# services/stress_behavior_service.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, List, Any
import threading

import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Encoders run on the BACKEND (not the browser)
try:
    import onnxruntime as ort
except Exception:
    ort = None

# -------------------- paths --------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts"
LAB = PROJECT_ROOT / "labels"
CAL_DIR = ART / "calibrators"

# Legacy global stack (kept for fallback only)
SCALER_PKL = ART / "global_head_scaler.joblib"                 # 17-D (MVP)
SCALER_PKL_HYB = ART / "global_head_scaler_hybrid.joblib"      # 145-D (HYBRID)
META_JSON  = ART / "global_head_meta.json"

# Encoders & normalization
ENC_MOUSE_ONNX = ART / "encoder_mouse(2).onnx"
ENC_KEYS_ONNX  = ART / "encoder_keystroke(2).onnx"
NORM_STATS_NPZ = ART / "norm_stats(2).npz"

# NEW: personal heads (preferred)
PERS_DIR = PROJECT_ROOT / "models_personal"
PERS_EMB = PERS_DIR / "personal_head_emb.joblib"
PERS_HYB = PERS_DIR / "personal_head_hybrid.joblib"
PERS_META = PERS_DIR / "personal_head_meta.json"

_LOCK = threading.Lock()
_PREDICTOR = None    # type: Optional["BehaviorPredictor"]
_SMOOTHERS: Dict[str, "TemporalSmoother"] = {}

# -------------------- helpers --------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _nz_count(vec: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(vec)))

def _to_f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)

def _extract_features(row: Dict, feature_names: List[str]) -> np.ndarray:
    return np.array([float(row.get(n, 0.0) or 0.0) for n in feature_names], dtype=np.float32)[None, :]

def _is_idle(row: Dict, eps: float = 1e-6) -> bool:
    return (
        float(row.get("ks_event_count", 0.0) or 0.0) <= eps and
        float(row.get("ks_keydowns", 0.0) or 0.0) <= eps and
        float(row.get("ks_keyups", 0.0) or 0.0) <= eps and
        float(row.get("mouse_move_count", 0.0) or 0.0) <= eps and
        float(row.get("mouse_click_count", 0.0) or 0.0) <= eps and
        float(row.get("mouse_scroll_count", 0.0) or 0.0) <= eps and
        float(row.get("active_seconds_fraction", 0.0) or 0.0) <= 0.02
    )

def _load_norm_stats() -> dict:
    stats = {}
    if NORM_STATS_NPZ.exists():
        d = np.load(NORM_STATS_NPZ, allow_pickle=True)
        for k in d.files:
            stats[k] = d[k]
    return stats

def _z_norm(x: np.ndarray, mean: np.ndarray | float | None, std: np.ndarray | float | None, eps=1e-6) -> np.ndarray:
    if mean is None or std is None:
        return x
    return (x - mean) / (std + eps)

# -------------------- smoothing --------------------
class TemporalSmoother:
    def __init__(self, alpha_active=0.35, alpha_idle=0.85, on_thresh=0.60, off_thresh=0.40, idle_reset_k=2, baseline=0.20):
        self.alpha_active = float(alpha_active)
        self.alpha_idle   = float(alpha_idle)
        self.on_t = float(on_thresh)
        self.off_t = float(off_thresh)
        self.idle_reset_k = int(idle_reset_k)
        self.baseline = float(baseline)
        self._ema: float | None = None
        self._state = 0
        self._idle_count = 0
    def step(self, p: float, is_idle: bool) -> tuple[float, int]:
        a = self.alpha_idle if is_idle else self.alpha_active
        self._ema = p if self._ema is None else (a * p + (1 - a) * self._ema)
        if is_idle:
            self._idle_count += 1
            if self._idle_count >= self.idle_reset_k:
                self._ema = self.baseline
                self._state = 0
        else:
            self._idle_count = 0
        if self._state == 0 and self._ema >= self.on_t:
            self._state = 1
        elif self._state == 1 and self._ema <= self.off_t:
            self._state = 0
        return float(self._ema), int(self._state)
    def force_off(self):
        self._state = 0

# -------------------- Platt calibrator --------------------
class PlattCalibrator:
    def __init__(self, coef_: float | None = None, intercept_: float | None = None):
        self.coef_, self.intercept_ = coef_, intercept_
    def is_fit(self) -> bool:
        return (self.coef_ is not None) and (self.intercept_ is not None)
    def fit(self, scores: np.ndarray, y: np.ndarray):
        yb = (y >= 0.5).astype(int)
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(scores.reshape(-1, 1), yb)
        self.coef_ = float(lr.coef_[0, 0])
        self.intercept_ = float(lr.intercept_[0])
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        z = self.coef_ * scores + self.intercept_
        return 1.0 / (1.0 + np.exp(-z))
    def to_dict(self) -> dict:
        return {"coef_": self.coef_, "intercept_": self.intercept_}
    @classmethod
    def from_file(cls, path: Path) -> "PlattCalibrator":
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            d = json.load(f)
        return cls(d.get("coef_"), d.get("intercept_"))
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

# -------------------- ONNX encoders with shape adaptation --------------------
class _OnnxEncoder:
    """
    Wraps an ONNX encoder and adapts input [1, T, F] to what the model expects.
    Pads/truncates time and slices/pads feature dim if needed.
    """
    def __init__(self, path: Path, name: str):
        if ort is None:
            raise RuntimeError("onnxruntime not installed; cannot run embeddings.")
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX model at {path}")
        self.name = name
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        ishape = self.sess.get_inputs()[0].shape  # [N, T, F] style
        self.exp_T = ishape[1] if len(ishape) >= 3 else None
        self.exp_F = ishape[2] if len(ishape) >= 3 else None
        self.exp_T = int(self.exp_T) if isinstance(self.exp_T, (int, np.integer)) else None
        self.exp_F = int(self.exp_F) if isinstance(self.exp_F, (int, np.integer)) else None
        self.onames = [o.name for o in self.sess.get_outputs()]

        self.out_D = None
        try:
            oshape = self.sess.get_outputs()[0].shape
            if isinstance(oshape[-1], (int, np.integer)):
                self.out_D = int(oshape[-1])
        except Exception:
            self.out_D = None

    def _conform(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3 and x.shape[0] == 1, f"{self.name}: expected [1,T,F], got {x.shape}"
        T, F = x.shape[1], x.shape[2]
        # time adapt
        if self.exp_T is not None and T != self.exp_T:
            if T < self.exp_T:
                pad = np.zeros((1, self.exp_T - T, F), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, -self.exp_T:, :]
        # feature adapt
        if self.exp_F is not None and F != self.exp_F:
            if F > self.exp_F:
                x = x[:, :, : self.exp_F]
            else:
                pad = np.zeros((1, x.shape[1], self.exp_F - F), dtype=np.float32)
                x = np.concatenate([x, pad], axis=2)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = _to_f32(x)
        x = self._conform(x)
        y = self.sess.run(self.onames, {self.iname: x})[0]
        if y.ndim == 3:
            y = y.mean(axis=1, keepdims=False)
        return _to_f32(y)  # [1, D]

def _build_mouse_seq(row: Dict, stats: dict) -> Optional[np.ndarray]:
    """
    Preferred: row["mouse_events"] = [[t_ms, x, y, type], ...]
    Derived channels: dx, dy, dt, speed, accel, type01 -> [T, 6]
    """
    seq = None
    if isinstance(row.get("mouse_seq"), list) and len(row["mouse_seq"]) > 0:
        seq = np.asarray(row["mouse_seq"], dtype=np.float32)
    elif isinstance(row.get("mouse_events"), list) and len(row["mouse_events"]) > 1:
        ev = np.asarray(row["mouse_events"], dtype=np.float32)  # [T, 4]
        t, x, y = ev[:, 0], ev[:, 1], ev[:, 2]
        typ = ev[:, 3] if ev.shape[1] >= 4 else np.zeros_like(t)
        dt = np.clip(np.diff(t, prepend=t[0]), 1e-3, None)
        dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx**2 + dy**2) / dt
        accel = np.diff(speed, prepend=speed[0]) / dt
        type01 = np.clip(typ / 2.0, 0.0, 1.0)  # 0 move, 1 click, 2 scroll -> [0..1]
        seq = np.stack([dx, dy, dt, speed, accel, type01], axis=1)  # [T, 6]
    else:
        return None

    m_mean, m_std = stats.get("mouse_mean", None), stats.get("mouse_std", None)
    if isinstance(m_mean, np.ndarray) and m_mean.ndim == 1 and m_mean.shape[0] == seq.shape[1]:
        seq = _z_norm(seq, m_mean, m_std)
    return seq[None, :, :]  # [1, T, C]

def _build_key_seq(row: Dict, stats: dict) -> Optional[np.ndarray]:
    """
    Preferred: row["key_events"] = [{down_ts, up_ts, next_down_ts}, ...]
    Derived channels: dwell_ms, flight_ms, ikg_ms -> [T,3]
    """
    seq = None
    if isinstance(row.get("key_seq"), list) and len(row["key_seq"]) > 0:
        seq = np.asarray(row["key_seq"], dtype=np.float32)  # [T, F]
    elif isinstance(row.get("key_events"), list) and len(row["key_events"]) > 0:
        dwell, flight, ikg = [], [], []
        ke = row["key_events"]
        for i, ev in enumerate(ke):
            kd = float(ev.get("down_ts", 0.0) or 0.0)
            ku = float(ev.get("up_ts", kd) or kd)
            nd = float(ev.get("next_down_ts", ku) or ku)
            dwell.append(max(ku - kd, 0.0))
            flight.append(max(nd - ku, 0.0))
            prev_kd = float(ke[i-1].get("down_ts", kd) if i > 0 else kd)
            ikg.append(max(kd - prev_kd, 0.0))
        seq = np.stack([np.array(dwell), np.array(flight), np.array(ikg)], axis=1)  # [T,3]
    else:
        return None

    k_mean, k_std = stats.get("key_mean", None), stats.get("key_std", None)
    if isinstance(k_mean, np.ndarray) and k_mean.ndim == 1 and k_mean.shape[0] == seq.shape[1]:
        seq = _z_norm(seq, k_mean, k_std)
    return seq[None, :, :]  # [1, T, C]

# -------------------- Predictor --------------------
class BehaviorPredictor:
    def __init__(self):
        # Personal meta (preferred)
        self.pmeta = _load_json(PERS_META)
        # Legacy meta (for feature_names + thresholds, if needed)
        self.gmeta = _load_json(META_JSON)

        # MVP feature names (from legacy meta or personal meta fallbacks)
        self.feature_names: List[str] = list(
            (self.gmeta.get("features") or self.gmeta.get("feature_names") or
             self.pmeta.get("mvp_cols") or [])
        )
        # Masks from personal meta (if present)
        self.mask_cols: List[str] = list(self.pmeta.get("mask_cols") or [])
        # Column orders for personal heads
        self.emb_cols_order: List[str] = list(self.pmeta.get("emb_cols") or [])
        self.hybrid_cols_order: List[str] = list(self.pmeta.get("hybrid_cols") or [])

        # encoders
        if ort is None or not (ENC_MOUSE_ONNX.exists() and ENC_KEYS_ONNX.exists()):
            raise RuntimeError("ONNX encoders not available; cannot run embeddings.")
        self.enc_mouse = _OnnxEncoder(ENC_MOUSE_ONNX, "mouse")
        self.enc_keys  = _OnnxEncoder(ENC_KEYS_ONNX, "keyboard")
        self.dim_m = int(self.enc_mouse.out_D or 64)
        self.dim_k = int(self.enc_keys.out_D or 64)
        self.norm_stats = _load_norm_stats()

        # Load personal heads (preferred)
        self.head_emb = joblib.load(PERS_EMB) if PERS_EMB.exists() else None
        self.head_hyb = joblib.load(PERS_HYB) if PERS_HYB.exists() else None
        self.has_personal = (self.head_emb is not None) or (self.head_hyb is not None)

        # Legacy hybrid stack (fallback only)
        self.scaler_hyb = joblib.load(SCALER_PKL_HYB) if SCALER_PKL_HYB.exists() else None
        legacy_mode = (self.gmeta.get("mode") or "regression")
        legacy_head = ART / f"global_head_{legacy_mode}_hybrid.joblib"
        self.legacy_head_hyb = joblib.load(legacy_head) if legacy_head.exists() else None

        # thresholds/hysteresis
        self.default_thresh = float(self.pmeta.get("thresholds", {}).get("hyb_best_f1_thr",
                                   self.gmeta.get("best_thresh", 0.5)))
        self.alpha = 0.35
        self.on_delta = 0.10
        self.off_delta = 0.10
        self.idle_clamp = float(self.gmeta.get("idle_clamp_prob", 0.10))

    # ---------- input builders ----------
    def _build_emb_pair(self, row: Dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        mseq = _build_mouse_seq(row, self.norm_stats)
        kseq = _build_key_seq(row, self.norm_stats)
        emb_m = self.enc_mouse(mseq) if mseq is not None else None
        emb_k = self.enc_keys(kseq)  if kseq is not None else None
        return emb_m, emb_k

    def _build_emb_vector(self, emb_m: np.ndarray | None, emb_k: np.ndarray | None) -> np.ndarray:
        if emb_m is None and emb_k is None:
            raise ValueError("No mouse/key sequences present for embeddings.")
        if emb_m is None:
            emb_m = np.zeros((1, self.dim_m), dtype=np.float32)
        if emb_k is None:
            emb_k = np.zeros((1, self.dim_k), dtype=np.float32)
        return np.concatenate([emb_m, emb_k], axis=1)  # [1, dim_m + dim_k]

    # ---------- heads ----------
    def _prob_from_model(self, model, X: np.ndarray) -> float:
        # For regressors (GBRT): clip into [0,1]; for classifiers: use proba if available
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0, 1])
        return float(np.clip(model.predict(X)[0], 0.0, 1.0))

    def _predict_personal(self, row: Dict, which: str | None) -> tuple[float, str]:
        """
        Returns (prob, head_used) using personal heads.
        which: 'emb' | 'hybrid' | None (auto)
        """
        emb_m, emb_k = self._build_emb_pair(row)
        has_any_emb = (emb_m is not None) or (emb_k is not None)

        # Build emb vector
        if has_any_emb:
            emb = self._build_emb_vector(emb_m, emb_k)  # [1, D_emb]
        else:
            emb = None

        # Determine head
        if which in ("emb", "hybrid"):
            force = which
        else:
            # auto choose hybrid if we have features for it and any emb
            force = "hybrid" if (has_any_emb and self.head_hyb is not None) else "emb"

        if force == "emb":
            if self.head_emb is None or emb is None:
                raise ValueError("Embeddings-only head not available or no embeddings provided.")
            # Build feature row in meta order if provided (optional for emb-only)
            X = emb
            prob = self._prob_from_model(self.head_emb, X)
            return prob, "emb"

        # hybrid
        if self.head_hyb is None or emb is None:
            raise ValueError("Hybrid head not available or no embeddings provided.")

        # 17 features and optional masks
        x17 = _extract_features(row, self.feature_names)  # [1, 17] (missing -> 0)
        masks = np.array([float(row.get(c, 0.0) or 0.0) for c in self.mask_cols], dtype=np.float32)[None, :] if self.mask_cols else None
        if masks is not None and masks.shape[1] > 0:
            X = np.concatenate([emb, x17, masks], axis=1)
        else:
            X = np.concatenate([emb, x17], axis=1)

        # If a strict column order is present in meta, reindex to that order
        if self.hybrid_cols_order:
            # Build a dict to reconstruct DataFrame row safely
            # We name emb dims as they appear in meta (e.g., 'mouse_emb_0', 'key_emb_0', etc.)
            # But personal meta already stores the order string list; we can map in sequence.
            row_map: Dict[str, float] = {}
            # First emb dims (mouse+key concatenated) — we fill in the same order the model was trained with
            for i, col in enumerate(self.hybrid_cols_order):
                if col.startswith("mouse_emb_") or col.startswith("key_emb_"):
                    row_map[col] = float(X[0, i])  # temp fill; will overwrite below
            # Easier approach: rebuild X by name using meta
            # Build arrays for name->value
            # Create name vector sequentially: emb part then mvp then masks, matching lengths
            values: List[float] = []
            # 1) Emb part (we don't have separate names here; rely on counts)
            emb_len = emb.shape[1]
            emb_vals = emb.flatten().tolist()
            # 2) MVP part
            mvp_vals = x17.flatten().tolist()
            # 3) mask part
            mask_vals = masks.flatten().tolist() if masks is not None else []
            # Concatenate in the same order we built X above
            built_vals = emb_vals + mvp_vals + mask_vals

            # Now place values into order by iterating hybrid_cols_order sequentially
            # (assumes hybrid_cols_order length equals X width)
            if len(self.hybrid_cols_order) == len(built_vals):
                X = np.asarray(built_vals, dtype=np.float32)[None, :]
            # else keep X as is

        prob = self._prob_from_model(self.head_hyb, X)
        return prob, "hybrid"

    def _predict_legacy(self, row: Dict) -> float:
        """
        Fallback: use legacy global hybrid stack if personal heads missing.
        """
        if self.legacy_head_hyb is None or self.scaler_hyb is None:
            raise RuntimeError("Legacy hybrid head/scaler not available.")
        emb_m, emb_k = self._build_emb_pair(row)
        emb = self._build_emb_vector(emb_m, emb_k)
        x17 = _extract_features(row, self.feature_names)
        X = np.concatenate([emb, x17], axis=1)
        Xz = self.scaler_hyb.transform(X)
        if hasattr(self.legacy_head_hyb, "predict_proba"):
            return float(self.legacy_head_hyb.predict_proba(Xz)[0, 1])
        return float(np.clip(self.legacy_head_hyb.predict(Xz)[0], 0.0, 1.0))

    # ---------- public predict ----------
    def predict(self, row: Dict, smoother: Optional[TemporalSmoother] = None) -> Dict:
        user_pref = str(row.get("head") or "").strip().lower() or None  # 'emb' | 'hybrid' | None
        if self.has_personal:
            raw_prob, head_used = self._predict_personal(row, which=user_pref)
        else:
            raw_prob = self._predict_legacy(row)
            head_used = "legacy_hybrid"

        # per-user Platt calibration (optional)
        user_id = str(row.get("user_id", "harsh"))
        cal = load_user_calibrator(user_id)
        if cal.is_fit():
            cal_prob = float(cal.predict_proba(np.array([raw_prob]))[0])
            has_cal = True
        else:
            cal_prob = raw_prob
            has_cal = False

        # idle clamp
        idle = _is_idle(row)
        if idle:
            cal_prob = min(cal_prob, self.idle_clamp)

        # smoothing + hysteresis
        if smoother is None:
            smoother = TemporalSmoother(
                alpha_active=self.alpha, alpha_idle=0.85,
                on_thresh=self.default_thresh + self.on_delta,
                off_thresh=self.default_thresh - self.off_delta,
                idle_reset_k=2, baseline=0.20,
            )
        smoothed, is_on = smoother.step(cal_prob, is_idle=idle)
        if idle:
            smoother.force_off()
            is_on = 0

        return {
            "raw_prob": raw_prob,
            "calibrated_prob": cal_prob,
            "smoothed_prob": smoothed,
            "is_stressed": bool(is_on),
            "threshold_used": self.default_thresh,
            "has_calibrator": has_cal,
            "input_kind": ("hybrid_embed" if head_used.startswith("hyb") or head_used == "legacy_hybrid" else "emb_only"),
            "head_used": head_used,
        }

# -------------------- module API --------------------
def init_service() -> None:
    global _PREDICTOR
    with _LOCK:
        if _PREDICTOR is None:
            _PREDICTOR = BehaviorPredictor()

def get_predictor() -> BehaviorPredictor:
    if _PREDICTOR is None:
        init_service()
    return _PREDICTOR

def load_user_calibrator(user_id: str) -> PlattCalibrator:
    path = CAL_DIR / f"cal_{user_id}.json"
    return PlattCalibrator.from_file(path)

def health_check() -> Dict:
    try:
        pmeta = _load_json(PERS_META)
        gmeta = _load_json(META_JSON)
        return {
            "ok": True,
            "personal_models_dir": str(PERS_DIR),
            "has_personal_emb": PERS_EMB.exists(),
            "has_personal_hybrid": PERS_HYB.exists(),
            "personal_meta_keys": list(pmeta.keys()) if pmeta else [],
            "declared_input_kind": str(pmeta.get("hybrid_cols", "hybrid_embed") if pmeta else (gmeta.get("input_kind") or "hybrid_embed")),
            "features": (gmeta.get("features") or gmeta.get("feature_names") or pmeta.get("mvp_cols") or []),
            "has_encoders": bool(ENC_MOUSE_ONNX.exists() and ENC_KEYS_ONNX.exists() and ort is not None),
            "has_norm_stats": bool(NORM_STATS_NPZ.exists()),
            "has_legacy_hybrid": bool(SCALER_PKL_HYB.exists() and (ART / f"global_head_{gmeta.get('mode','regression')}_hybrid.joblib").exists()),
            "requires_sequences": True
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def latest_window_features(user_id: str | None = None) -> dict:
    pred = get_predictor()
    # Use your newer CSV if present; else fallback
    csv_path = (LAB / "stress_hybrid_10s.csv") if (LAB / "stress_hybrid_10s.csv").exists() else (LAB / "stress_30s.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV at {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    if len(df) == 0:
        raise ValueError("CSV empty.")
    last = df.iloc[-1].to_dict()
    out = {}
    for name in pred.feature_names:
        try:
            out[name] = float(last.get(name, 0.0) or 0.0)
        except Exception:
            out[name] = 0.0
    out["user_id"] = str(user_id or last.get("user_id") or "harsh")
    return out

def predict_from_row(row: Dict, user_id: Optional[str] = None) -> Dict:
    pred = get_predictor()
    uid = str(user_id or row.get("user_id") or "harsh")
    with _LOCK:
        smoother = _SMOOTHERS.get(uid)
        if smoother is None:
            smoother = TemporalSmoother(
                alpha_active=pred.alpha, alpha_idle=0.85,
                on_thresh=pred.default_thresh + pred.on_delta,
                off_thresh=pred.default_thresh - pred.off_delta,
                idle_reset_k=2, baseline=0.20,
            )
            _SMOOTHERS[uid] = smoother

    out = pred.predict({**row, "user_id": uid}, smoother)
    x17 = _extract_features(row, pred.feature_names)
    return {
        "user_id": uid,
        **out,
        "feature_count": len(pred.feature_names),
        "nonzero_features": _nz_count(x17),
        "activity": {
            "ks_event_count": float(row.get("ks_event_count", 0.0) or 0.0),
            "ks_keydowns": float(row.get("ks_keydowns", 0.0) or 0.0),
            "ks_keyups": float(row.get("ks_keyups", 0.0) or 0.0),
            "mouse_move_count": float(row.get("mouse_move_count", 0.0) or 0.0),
            "mouse_click_count": float(row.get("mouse_click_count", 0.0) or 0.0),
            "mouse_scroll_count": float(row.get("mouse_scroll_count", 0.0) or 0.0),
            "active_seconds_fraction": float(row.get("active_seconds_fraction", 0.0) or 0.0),
        },
    }
