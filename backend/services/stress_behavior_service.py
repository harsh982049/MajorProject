from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import threading

import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# NEW: ONNX runtime for encoders
try:
    import onnxruntime as ort
except Exception:
    ort = None  # we will gracefully fall back if unavailable

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART = PROJECT_ROOT / "artifacts"
LAB = PROJECT_ROOT / "labels"
CAL_DIR = ART / "calibrators"

SCALER_PKL = ART / "global_head_scaler.joblib"               # MVP scaler (17 features)
SCALER_PKL_HYB = ART / "global_head_scaler_hybrid.joblib"    # NEW (hybrid) scaler if you train one
META_JSON  = ART / "global_head_meta.json"

# Encoders + stats
ENC_MOUSE_ONNX = ART / "encoder_mouse.onnx"
ENC_KEYS_ONNX  = ART / "encoder_keystroke.onnx"
NORM_STATS_NPZ = ART / "norm_stats.npz"  # must include means/stds for mouse/key channels

# ---------- Internal singletons ----------
_LOCK = threading.Lock()
_PREDICTOR = None  # type: Optional[BehaviorPredictor]
_SMOOTHERS: Dict[str, "TemporalSmoother"] = {}  # per-user smoothing state


# ---------- Utilities ----------
def _load_meta() -> dict:
    if not META_JSON.exists():
        raise FileNotFoundError(f"Missing meta JSON at {META_JSON}")
    with open(META_JSON, "r") as f:
        return json.load(f)

def _autodiscover_head(meta: dict, hybrid: bool) -> Path:
    """
    For 17-feature MVP:
      global_head_{mode}_behavior.joblib
    For hybrid (embeddings+features):
      global_head_{mode}_hybrid.joblib  (recommended naming)
      or fallback to global_head_{mode}_behavior.joblib if not found.
    """
    mode = meta.get("mode", "regression")
    if hybrid:
        preferred = ART / f"global_head_{mode}_hybrid.joblib"
        if preferred.exists():
            return preferred
    # fallback to MVP head
    fallback = ART / f"global_head_{mode}_behavior.joblib"
    if fallback.exists():
        return fallback
    # any global_head_*.joblib as last resort
    cands = sorted(ART.glob("global_head_*.joblib"))
    if not cands:
        raise FileNotFoundError(f"No global_head_*.joblib found in {ART}")
    return cands[-1]

def _extract_features(row: Dict, feature_names) -> np.ndarray:
    """Build a [1, D] feature vector; missing fields default to 0.0."""
    return np.array([float(row.get(name, 0.0) or 0.0) for name in feature_names], dtype=np.float32)[None, :]

def _nz_count(vec: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(vec)))

def _to_float32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)

# ---------- Activity-aware EMA + hysteresis ----------
class TemporalSmoother:
    def __init__(
        self,
        alpha_active: float = 0.35,
        alpha_idle: float   = 0.85,
        on_thresh: float    = 0.60,
        off_thresh: float   = 0.40,
        idle_reset_k: int   = 2,
        baseline: float     = 0.20,
    ):
        self.alpha_active = float(alpha_active)
        self.alpha_idle   = float(alpha_idle)
        self.on_t   = float(on_thresh)
        self.off_t  = float(off_thresh)
        self.idle_reset_k = int(idle_reset_k)
        self.baseline = float(baseline)

        self._ema: float | None = None
        self._state = 0
        self._idle_count = 0

    def step(self, p: float, is_idle: bool) -> tuple[float, int]:
        a = self.alpha_idle if is_idle else self.alpha_active
        if self._ema is None:
            self._ema = p
        else:
            self._ema = a * p + (1 - a) * self._ema

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

    def force_off(self) -> None:
        self._state = 0


# ---------- Per-user Platt calibrator ----------
class PlattCalibrator:
    def __init__(self, coef_: Optional[float] = None, intercept_: Optional[float] = None):
        self.coef_ = coef_
        self.intercept_ = intercept_

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


# ---------- Embedding encoders (NEW) ----------
class _OnnxEncoder:
    def __init__(self, path: Path, providers: Optional[List[str]] = None):
        if ort is None:
            raise RuntimeError("onnxruntime not installed; cannot use embeddings.")
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX model at {path}")
        self.sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: [1, T, F] float32
        out = self.sess.run(self.onames, {self.iname: _to_float32(x)})
        y = out[0]
        # supports either [1, D] or [1, T, D] with pooling inside model
        if y.ndim == 3:
            y = y.mean(axis=1, keepdims=False)
        return _to_float32(y)

def _load_norm_stats() -> dict:
    stats = {}
    if NORM_STATS_NPZ.exists():
        d = np.load(NORM_STATS_NPZ, allow_pickle=True)
        for k in d.files:
            stats[k] = d[k]
    return stats

def _z_norm(x: np.ndarray, mean: np.ndarray | float | None, std: np.ndarray | float | None, eps: float = 1e-6) -> np.ndarray:
    if mean is None or std is None:
        return x
    return (x - mean) / (std + eps)

def _build_mouse_seq(row: Dict, stats: dict) -> Optional[np.ndarray]:
    """
    Expect either:
      - row["mouse_seq"] = [[dx, dy, dt, speed, accel], ...]  (preferred)
      - or row["mouse_events"] = [[t, x, y, type], ...] from which we derive the 5 channels.
    Returns [1, T, 5] or None if not enough data.
    """
    if "mouse_seq" in row and isinstance(row["mouse_seq"], list) and len(row["mouse_seq"]) > 0:
        seq = np.asarray(row["mouse_seq"], dtype=np.float32)
    elif "mouse_events" in row and isinstance(row["mouse_events"], list) and len(row["mouse_events"]) > 1:
        ev = np.asarray(row["mouse_events"], dtype=np.float32)  # shape [T, 4] -> t,x,y,type
        t, x, y = ev[:, 0], ev[:, 1], ev[:, 2]
        dt = np.clip(np.diff(t, prepend=t[0]), 1e-3, None)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx**2 + dy**2) / dt
        accel = np.diff(speed, prepend=speed[0]) / dt
        seq = np.stack([dx, dy, dt, speed, accel], axis=1)
    else:
        return None

    # z-norm
    m_mean = stats.get("mouse_mean", None)
    m_std  = stats.get("mouse_std", None)
    if isinstance(m_mean, np.ndarray) and m_mean.ndim == 1 and m_mean.shape[0] == seq.shape[1]:
        seq = _z_norm(seq, m_mean, m_std)
    return seq[None, :, :]  # [1, T, 5]

def _build_key_seq(row: Dict, stats: dict) -> Optional[np.ndarray]:
    """
    Expect either:
      - row["key_seq"] = [[dwell, flight, delta_t], ...]
      - or row["key_events"] = [{"down_ts":..., "up_ts":..., "next_down_ts":...}, ...]
    Returns [1, E, F] or None if not enough data.
    """
    if "key_seq" in row and isinstance(row["key_seq"], list) and len(row["key_seq"]) > 0:
        seq = np.asarray(row["key_seq"], dtype=np.float32)
    elif "key_events" in row and isinstance(row["key_events"], list) and len(row["key_events"]) > 0:
        dwell, flight, dt = [], [], []
        for i, ev in enumerate(row["key_events"]):
            kd = float(ev.get("down_ts", 0.0) or 0.0)
            ku = float(ev.get("up_ts", kd) or kd)
            nd = float(ev.get("next_down_ts", ku) or ku)
            dwell.append(max(ku - kd, 0.0))
            flight.append(max(nd - ku, 0.0))
            dt.append(max(kd - float(row["key_events"][i-1].get("down_ts", kd) if i > 0 else kd), 0.0))
        seq = np.stack([np.array(dwell), np.array(flight), np.array(dt)], axis=1)
    else:
        return None

    k_mean = stats.get("key_mean", None)
    k_std  = stats.get("key_std", None)
    if isinstance(k_mean, np.ndarray) and k_mean.ndim == 1 and k_mean.shape[0] == seq.shape[1]:
        seq = _z_norm(seq, k_mean, k_std)
    return seq[None, :, :]  # [1, E, F]


# ---------- Core predictor ----------
def _is_idle(row: Dict, eps: float = 1e-6) -> bool:
    ks_events = float(row.get("ks_event_count", 0.0) or 0.0)
    kd = float(row.get("ks_keydowns", 0.0) or 0.0)
    ku = float(row.get("ks_keyups", 0.0) or 0.0)
    mm = float(row.get("mouse_move_count", 0.0) or 0.0)
    mc = float(row.get("mouse_click_count", 0.0) or 0.0)
    ms = float(row.get("mouse_scroll_count", 0.0) or 0.0)
    act = float(row.get("active_seconds_fraction", 0.0) or 0.0)
    return (ks_events <= eps and kd <= eps and ku <= eps and
            mm <= eps and mc <= eps and ms <= eps and
            act <= 0.02)

class BehaviorPredictor:
    def __init__(self):
        meta = _load_meta()
        self.meta = meta

        # NEW: interpret input kind
        # "behavior_csv_only" (your current meta) -> use 17 features only
        # "hybrid_embed" -> expect encoders + sequences and concatenate [embeddings ⊕ 17 features]
        # "embed_only"   -> use embeddings only (no 17s)
        self.input_kind = str(meta.get("input_kind") or "behavior_csv_only")

        # MVP 17-feature names
        self.feature_names = meta.get("feature_names") or []
        if not isinstance(self.feature_names, list):
            raise ValueError("Invalid 'feature_names' in meta.")

        # Try to load encoders if hybrid/embed_only
        self.enc_mouse = None
        self.enc_keys = None
        self.norm_stats = {}
        if self.input_kind in ("hybrid_embed", "embed_only"):
            self.norm_stats = _load_norm_stats()
            if ort is not None and ENC_MOUSE_ONNX.exists() and ENC_KEYS_ONNX.exists():
                self.enc_mouse = _OnnxEncoder(ENC_MOUSE_ONNX)
                self.enc_keys  = _OnnxEncoder(ENC_KEYS_ONNX)
            else:
                # If encoders not available, quietly fall back to 17-feature path
                self.input_kind = "behavior_csv_only"

        # Decide which scaler/head to load
        hybrid = (self.input_kind == "hybrid_embed")
        head_path = _autodiscover_head(meta, hybrid=hybrid)

        # Choose scaler
        scaler_path = SCALER_PKL_HYB if hybrid and SCALER_PKL_HYB.exists() else SCALER_PKL
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler at {scaler_path}")

        self.scaler = joblib.load(scaler_path)
        self.head   = joblib.load(head_path)

        # thresholds from training (used to set hysteresis)
        self.default_thresh = float(meta.get("best_thresh", 0.5))

        # Smoother defaults
        self.alpha = 0.35
        self.on_delta = 0.10
        self.off_delta = 0.10

        # Idle clamp configuration
        self.idle_clamp = float(meta.get("idle_clamp_prob", 0.10))

    def _head_prob(self, Xz: np.ndarray) -> float:
        if hasattr(self.head, "predict_proba"):
            return float(self.head.predict_proba(Xz)[0, 1])
        return float(np.clip(self.head.predict(Xz)[0], 0.0, 1.0))

    def _make_embedding(self, row: Dict) -> Optional[np.ndarray]:
        """Compute [mouse_embed ⊕ key_embed] from per-window sequences."""
        if self.enc_mouse is None or self.enc_keys is None:
            return None
        mseq = _build_mouse_seq(row, self.norm_stats)   # [1, T, 5]
        kseq = _build_key_seq(row, self.norm_stats)     # [1, E, F]
        if mseq is None and kseq is None:
            return None
        pieces = []
        if mseq is not None:
            pieces.append(self.enc_mouse(mseq))  # [1, Dm]
        if kseq is not None:
            pieces.append(self.enc_keys(kseq))   # [1, Dk]
        if not pieces:
            return None
        emb = np.concatenate(pieces, axis=1)  # [1, Dm+Dk]
        return _to_float32(emb)

    def _build_input_vector(self, row: Dict) -> np.ndarray:
        """
        Returns X =   [1, D] for the model, depending on input_kind:
          - behavior_csv_only: 17-feature vector
          - embed_only:        embeddings only
          - hybrid_embed:      [embeddings ⊕ 17 features]
        If embeddings are not available for this row (missing sequences),
        we gracefully fall back to the 17-feature vector.
        """
        x_feat = _extract_features(row, self.feature_names) if self.feature_names else None
        if self.input_kind == "behavior_csv_only":
            return x_feat

        emb = self._make_embedding(row)
        if emb is None:
            # fallback if no sequences
            return x_feat

        if self.input_kind == "embed_only":
            return emb

        # hybrid: concat
        if x_feat is None:
            return emb
        return np.concatenate([emb, x_feat], axis=1)

    def predict(self, row: Dict, smoother: Optional[TemporalSmoother] = None) -> Dict:
        X = self._build_input_vector(row)
        Xz = self.scaler.transform(X)
        raw_prob = self._head_prob(Xz)

        # Optional per-user calibration
        user_id = str(row.get("user_id", "harsh"))
        cal = load_user_calibrator(user_id)
        if cal.is_fit():
            cal_prob = float(cal.predict_proba(np.array([raw_prob]))[0])
            has_cal = True
        else:
            cal_prob = raw_prob
            has_cal = False

        # Idle guard
        idle = _is_idle(row)
        if idle:
            cal_prob = min(cal_prob, self.idle_clamp)

        # Temporal smoothing + hysteresis
        if smoother is None:
            smoother = TemporalSmoother(
                alpha_active=self.alpha,
                alpha_idle=0.85,
                on_thresh=self.default_thresh + self.on_delta,
                off_thresh=self.default_thresh - self.off_delta,
                idle_reset_k=2,
                baseline=0.20,
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
            "input_kind": self.input_kind,
        }


# ---------- Service lifecycle ----------
def init_service() -> None:
    global _PREDICTOR
    with _LOCK:
        if _PREDICTOR is None:
            _PREDICTOR = BehaviorPredictor()

def get_predictor() -> BehaviorPredictor:
    if _PREDICTOR is None:
        init_service()
    return _PREDICTOR


# ---------- Public API for app.py ----------
def health_check() -> Dict:
    try:
        meta = _load_meta()
        # derive effective kind given artifact availability
        effective_kind = str(meta.get("input_kind") or "behavior_csv_only")
        if effective_kind in ("hybrid_embed", "embed_only"):
            if ort is None or not (ENC_MOUSE_ONNX.exists() and ENC_KEYS_ONNX.exists()):
                effective_kind = "behavior_csv_only"

        mode = meta.get("mode")
        head_path = _autodiscover_head(meta, hybrid=(effective_kind == "hybrid_embed"))

        on_t = float(meta.get("best_thresh", 0.5)) + 0.10
        off_t = float(meta.get("best_thresh", 0.5)) - 0.10

        return {
            "ok": True,
            "mode": mode,
            "input_kind": effective_kind,
            "features": meta.get("feature_names"),
            "artifacts_dir": str(ART),
            "labels_dir": str(LAB),
            "has_encoders": bool(ENC_MOUSE_ONNX.exists() and ENC_KEYS_ONNX.exists()),
            "has_norm_stats": bool(NORM_STATS_NPZ.exists()),
            "head_used": str(head_path.name),
            "smoother": {"alpha": 0.35, "on": on_t, "off": off_t},
            "idle_clamp_prob": float(meta.get("idle_clamp_prob", 0.10)),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def predict_from_row(row: Dict, user_id: Optional[str] = None) -> Dict:
    pred = get_predictor()
    uid = str(user_id or row.get("user_id") or "harsh")
    with _LOCK:
        smoother = _SMOOTHERS.get(uid)
        if smoother is None:
            smoother = TemporalSmoother(
                alpha_active=pred.alpha,
                alpha_idle=0.85,
                on_thresh=pred.default_thresh + pred.on_delta,
                off_thresh=pred.default_thresh - pred.off_delta,
                idle_reset_k=2,
                baseline=0.20,
            )
            _SMOOTHERS[uid] = smoother

    # Build MVP features in trained order for activity debug
    x = _extract_features(row, pred.feature_names) if pred.feature_names else np.zeros((1, 0), dtype=np.float32)
    out = pred.predict({**row, "user_id": uid}, smoother)

    activity_summary = {
        "ks_event_count": float(row.get("ks_event_count", 0.0) or 0.0),
        "ks_keydowns": float(row.get("ks_keydowns", 0.0) or 0.0),
        "ks_keyups": float(row.get("ks_keyups", 0.0) or 0.0),
        "mouse_move_count": float(row.get("mouse_move_count", 0.0) or 0.0),
        "mouse_click_count": float(row.get("mouse_click_count", 0.0) or 0.0),
        "mouse_scroll_count": float(row.get("mouse_scroll_count", 0.0) or 0.0),
        "active_seconds_fraction": float(row.get("active_seconds_fraction", 0.0) or 0.0),
    }

    return {
        "user_id": uid,
        **out,
        "feature_count": len(pred.feature_names),
        "nonzero_features": _nz_count(x),
        "activity": activity_summary,
    }

def latest_window_features(user_id: str | None = None) -> dict:
    init_service()
    pred = get_predictor()
    csv_path = LAB / "stress_30s.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found at {csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("CSV is empty.")
    last = df.iloc[-1].to_dict()

    out = {}
    for name in pred.feature_names:
        try:
            out[name] = float(last.get(name, 0.0) or 0.0)
        except Exception:
            out[name] = 0.0
    uid = (user_id or last.get("user_id") or "harsh")
    out["user_id"] = str(uid)
    return out

# ---------- Calibrator helpers ----------
def load_user_calibrator(user_id: str) -> PlattCalibrator:
    path = CAL_DIR / f"cal_{user_id}.json"
    return PlattCalibrator.from_file(path)

def train_user_calibrator(user_id: str = "harsh", min_rows: int = 200) -> str:
    init_service()
    pred = get_predictor()

    csv_path = LAB / "stress_30s.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if "user_id" in df.columns:
        df = df[df["user_id"].astype(str) == str(user_id)]
    if {"confident", "coverage"}.issubset(df.columns):
        df = df[(df["confident"] == 1) & (df["coverage"] >= pred.meta.get("conf_coverage_min", 0.30))]
    df = df.dropna(subset=["stress_prob"])
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows for calibration; found {len(df)}.")

    # IMPORTANT: calibrator training mirrors the head's current input space.
    # Here we recompute X the same way predict() does for each row.
    X_rows = []
    for _, r in df.iterrows():
        row = {k: r.get(k) for k in pred.feature_names}
        row = {**row, "user_id": user_id}  # ensure user_id
        X_rows.append(_extract_features(row, pred.feature_names)[0])
    X = np.vstack(X_rows) if X_rows else np.zeros((0, len(pred.feature_names)), dtype=np.float32)
    Xz = pred.scaler.transform(X)

    if hasattr(pred.head, "predict_proba"):
        scores = pred.head.predict_proba(Xz)[:, 1]
    else:
        scores = np.clip(pred.head.predict(Xz), 0, 1)
    y = df["stress_prob"].values.astype(np.float32)

    cal = PlattCalibrator()
    cal.fit(scores, y)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAL_DIR / f"cal_{user_id}.json"
    cal.save(out_path)
    return str(out_path)
