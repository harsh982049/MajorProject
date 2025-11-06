# face.py — Hybrid data logger (Face ONNX + Keystroke/Mouse → 17 features + embeddings → CSV)
# Requires: onnxruntime, opencv-python(-headless), numpy, pynput (optional), pyautogui (optional)

import os, time, csv, argparse, threading
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple

import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception as e:
    ort = None
    print("[WARN] onnxruntime not available; embeddings/face model will not run.")

# optional listeners
try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = None
    mouse = None
    print("[WARN] pynput not installed. Keyboard/mouse fields will stay zero.")

# ---------------- CLI ----------------
def get_args():
    ap = argparse.ArgumentParser(description="Face+Behavior hybrid logger → CSV (overlapping time windows)")
    # Face model
    default_face = os.path.abspath(os.path.join(os.path.dirname(__file__), "export", "face_emotion_pt.onnx"))
    ap.add_argument("--emotion_onnx", type=str, default=os.getenv("EMOTION_ONNX_PATH", default_face),
                    help="Path to face emotion ONNX (same as service).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--display", action="store_true")

    # Encoders / artifacts
    default_art = os.path.abspath(os.path.join(os.path.dirname(__file__), "artifacts"))
    ap.add_argument("--artifacts_dir", type=str, default=default_art,
                    help="Directory with encoder_mouse.onnx, encoder_keystroke.onnx, norm_stats.npz")

    # Output CSV (10s / 5s hop by default)
    ap.add_argument("--csv", type=str, default=os.path.join("labels", "stress_hybrid_10s.csv"))
    ap.add_argument("--window_sec", type=int, default=10, help="Window length in seconds (default 10s)")
    ap.add_argument("--hop_sec", type=int, default=5, help="Stride between windows in seconds (default 5s overlap)")
    ap.add_argument("--min_face_coverage", type=float, default=0.30)

    # Identity
    ap.add_argument("--user_id", type=str, default="harsh")
    ap.add_argument("--session_id", type=str, default=None)

    # Optional mouse polling to capture idle pointer movement (helps when OS throttles move events)
    ap.add_argument("--poll_hz", type=float, default=0.0)

    return ap.parse_args()

# ---------------- Face model config ----------------
CLASS_NAMES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutrality", "sadness", "surprise"
]
NO_ANXIETY_IDX = {CLASS_NAMES.index("happiness"), CLASS_NAMES.index("contempt"), CLASS_NAMES.index("neutrality")}
ANXIETY_IDX = set(range(len(CLASS_NAMES))) - NO_ANXIETY_IDX
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / x.size

def _detect_layout(shape: List[Any]) -> Tuple[bool, int, int, int]:
    # Return (is_nchw, C, H, W). Falls back if dynamic.
    is_nchw, C, H, W = True, 3, 224, 224
    if not isinstance(shape, list) or len(shape) != 4:
        return is_nchw, C, H, W
    n, d1, d2, d3 = shape
    def _to_int(v, default):
        try:
            return int(v) if v is not None else default
        except Exception:
            return default
    d1i, d2i, d3i = _to_int(d1, None), _to_int(d2, None), _to_int(d3, None)
    if d1i in (1, 3): return True, (d1i or C), (d2i or H), (d3i or W)
    if d3i in (1, 3): return False, (d3i or C), (d1i or H), (d2i or W)
    return is_nchw, C, H, W

def _enhance_gray(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return filtered

def _prep_tensor(enhanced_gray: np.ndarray, layout_nchw: bool, channels: int, h: int, w: int) -> np.ndarray:
    if channels == 1:
        arr = cv2.resize(enhanced_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        if layout_nchw:
            x = arr[None, None, :, :]
        else:
            x = arr[:, :, None][None, ...]
        return x.astype(np.float32)
    rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    if layout_nchw:
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
    else:
        x = rgb[None, ...]
    return x.astype(np.float32)

# ---------------- ONNX encoder wrapper (adapts [1,T,F] to model shape) ----------------
class _OnnxEncoder:
    def __init__(self, path: Path, name: str):
        if ort is None:
            raise RuntimeError("onnxruntime not installed; cannot run embeddings.")
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX model at {path}")
        self.name = name
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]; out = self.sess.get_outputs()[0]
        self.iname = inp.name; self.onames = [out.name]
        ishape = inp.shape
        self.exp_T = ishape[1] if len(ishape) >= 3 else None
        self.exp_F = ishape[2] if len(ishape) >= 3 else None
        self.exp_T = int(self.exp_T) if isinstance(self.exp_T, (int, np.integer)) else None
        self.exp_F = int(self.exp_F) if isinstance(self.exp_F, (int, np.integer)) else None
        # detect embedding width
        self.out_D = None
        try:
            oshape = out.shape
            if isinstance(oshape[-1], (int, np.integer)):
                self.out_D = int(oshape[-1])
        except Exception:
            self.out_D = None

    def _conform(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3 and x.shape[0] == 1, f"{self.name}: expected [1,T,F], got {x.shape}"
        T, F = x.shape[1], x.shape[2]
        if self.exp_T is not None and T != self.exp_T:
            if T < self.exp_T:
                pad = np.repeat(x[:, -1:, :], repeats=self.exp_T - T, axis=1)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, -self.exp_T:, :]
        if self.exp_F is not None and F != self.exp_F:
            if F > self.exp_F:
                x = x[:, :, : self.exp_F]
            else:
                pad = np.zeros((1, x.shape[1], self.exp_F - F), dtype=np.float32)
                x = np.concatenate([x, pad], axis=2)
        return x.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        T_valid = x.shape[1]  # before conform
        x = self._conform(x)
        y = self.sess.run(self.onames, {self.iname: x})[0]
        if y.ndim == 3:  # [1, T', D] → mean pool
            y = y.mean(axis=1, keepdims=False)
        # compensate for dilution when we padded
        if self.exp_T is not None and T_valid is not None and T_valid > 0 and T_valid < (self.exp_T):
            y = y * (self.exp_T / float(T_valid))
        return y.astype(np.float32)

# ---------------- Behavior aggregator → 17 MVP features ----------------
class BehaviorAggregator:
    def __init__(self):
        self.lock = threading.Lock()
        # keyboard
        self.keydown_times = {}
        self.dwell_times = []
        self.ikg_times = []
        self.ks_event_count = 0
        self.ks_keydowns = 0
        self.ks_keyups = 0
        self.ks_keys_seen = set()
        self.last_keydown_time = None
        # mouse
        self.mouse_move_count = 0
        self.mouse_click_count = 0
        self.mouse_scroll_count = 0
        self.mouse_total_distance = 0.0
        self.mouse_last_pos = None
        self.mouse_last_time = None
        self.mouse_speeds = []
        # activity coverage
        self.active_seconds = set()
        # raw sequences for embeddings
        self.mouse_events = []   # list of (t_ms, x, y, type) type:0 move,1 click,2 scroll
        self.key_events   = []   # list of dicts: {down_ts, up_ts, next_down_ts}

        self._last_key_down_ev = None  # temp holder to fill 'up_ts'

    def reset(self):
        with self.lock:
            self.__init__()

    def _mark(self, t):
        self.active_seconds.add(int(t))

    # ---------- keyboard ----------
    def on_keydown(self, k, t):
        with self.lock:
            self.ks_event_count += 1
            self.ks_keydowns += 1
            self.ks_keys_seen.add(str(k))
            if self.last_keydown_time is not None:
                self.ikg_times.append((t - self.last_keydown_time) * 1000.0)  # ms
            self.last_keydown_time = t
            self.keydown_times[str(k)] = t
            self._last_key_down_ev = {"down_ts": float(t), "up_ts": float(t)}  # will update on keyup
            self.key_events.append(self._last_key_down_ev)
            self._mark(t)

    def on_keyup(self, k, t):
        with self.lock:
            self.ks_event_count += 1
            self.ks_keyups += 1
            kk = str(k)
            if kk in self.keydown_times:
                dt = (t - self.keydown_times[kk]) * 1000.0  # ms
                if 0 <= dt < 5000:
                    self.dwell_times.append(dt)
                del self.keydown_times[kk]
            # update last key event's up_ts
            if self._last_key_down_ev is not None:
                self._last_key_down_ev["up_ts"] = float(t)
            self._mark(t)

    # ---------- mouse ----------
    def on_move(self, x, y, t):
        with self.lock:
            self.mouse_move_count += 1
            if self.mouse_last_pos is not None and self.mouse_last_time is not None:
                dx = x - self.mouse_last_pos[0]
                dy = y - self.mouse_last_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                dt = max(1e-3, t - self.mouse_last_time)
                self.mouse_total_distance += dist
                self.mouse_speeds.append(dist / dt)
            self.mouse_last_pos = (x, y)
            self.mouse_last_time = t
            self._mark(t)
            self.mouse_events.append((float(t) * 1000.0, float(x), float(y), 0.0))  # type 0 move

    def on_click(self, x, y, button, pressed, t):
        with self.lock:
            if pressed:
                self.mouse_click_count += 1
                self._mark(t)
                self.mouse_events.append((float(t) * 1000.0, float(x), float(y), 1.0))  # type 1 click

    def on_scroll(self, x, y, dx, dy, t):
        with self.lock:
            self.mouse_scroll_count += 1
            self._mark(t)
            self.mouse_events.append((float(t) * 1000.0, float(x), float(y), 2.0))  # type 2 scroll

    # ---------- per-window summary ----------
    def summarize(self, t0, t1):
        """
        Compute all 17 features + raw sequences using ONLY events in [t0, t1).
        Keys are in seconds; mouse event times are stored in ms (convert before filtering).
        """
        with self.lock:
            # -------- filter KEY events to the window --------
            keys_win = [ev for ev in self.key_events if t0 <= float(ev.get("down_ts", 0.0)) < t1]
            ks_event_count = len(keys_win) * 2  # approx: each has down+up

            # per-event dwell/ikg within the window (ms for MVPs)
            dwell, ikg = [], []
            last_down_ts = None
            for i, ev in enumerate(keys_win):
                kd = float(ev.get("down_ts", 0.0) or 0.0)
                ku = float(ev.get("up_ts", kd) or kd)
                nd = float(keys_win[i+1]["down_ts"]) if i+1 < len(keys_win) else ku
                dwell.append(max(ku - kd, 0.0) * 1000.0)  # ms
                if last_down_ts is not None:
                    ikg.append(max(kd - last_down_ts, 0.0) * 1000.0)  # ms
                last_down_ts = kd

            def _stats_ms(a):
                if not a: return (0.0, 0.0, 0.0)
                a = np.asarray(a, dtype=np.float32)
                return float(np.mean(a)), float(np.median(a)), float(np.percentile(a, 95))

            mdw, mdw_med, mdw_p95 = _stats_ms(dwell)
            mikg, mikg_med, mikg_p95 = _stats_ms(ikg)

            ks_keydowns = len(keys_win)
            ks_keyups = len(keys_win)
            ks_unique_keys = 0  # (we don't keep keycodes here)

            # -------- filter MOUSE events to the window --------
            t0_ms, t1_ms = float(t0) * 1000.0, float(t1) * 1000.0
            mouse_win = [ev for ev in self.mouse_events if t0_ms <= float(ev[0]) < t1_ms]
            mouse_move_count = sum(1 for ev in mouse_win if int(ev[3]) == 0)
            mouse_click_count = sum(1 for ev in mouse_win if int(ev[3]) == 1)
            mouse_scroll_count = sum(1 for ev in mouse_win if int(ev[3]) == 2)

            # reconstruct speeds within the window
            mouse_total_distance = 0.0
            mouse_speeds = []
            prev = None
            for ev in mouse_win:
                t_ms, x, y, typ = ev
                if prev is not None:
                    dt = max(1e-3, (t_ms - prev[0]) / 1000.0)
                    dx = x - prev[1]; dy = y - prev[2]
                    dist = (dx*dx + dy*dy) ** 0.5
                    if int(typ) == 0:  # count distance only for move
                        mouse_total_distance += dist
                    mouse_speeds.append(dist / dt)
                prev = ev
            mean_speed = float(np.mean(mouse_speeds)) if mouse_speeds else 0.0
            max_speed  = float(np.max(mouse_speeds)) if mouse_speeds else 0.0

            # -------- activity coverage fraction in [t0,t1) --------
            active_frac = len([s for s in self.active_seconds if (t0 <= s < t1)]) / max(1.0, (t1 - t0))

            # -------- sequences for encoders (seconds-based for keys) --------
            keys_seq = []
            for i, ev in enumerate(keys_win):
                kd = float(ev.get("down_ts", 0.0) or 0.0)
                ku = float(ev.get("up_ts", kd) or kd)
                nd = float(keys_win[i+1]["down_ts"]) if i+1 < len(keys_win) else ku
                keys_seq.append({"down_ts": kd, "up_ts": ku, "next_down_ts": nd})

            return {
                # 17 MVP features (windowed)
                "ks_event_count": int(ks_event_count),
                "ks_keydowns": int(ks_keydowns),
                "ks_keyups": int(ks_keyups),
                "ks_unique_keys": int(ks_unique_keys),
                "ks_mean_dwell_ms": round(mdw, 3),
                "ks_median_dwell_ms": round(mdw_med, 3),
                "ks_p95_dwell_ms": round(mdw_p95, 3),
                "ks_mean_ikg_ms": round(mikg, 3),
                "ks_median_ikg_ms": round(mikg_med, 3),
                "ks_p95_ikg_ms": round(mikg_p95, 3),
                "mouse_move_count": int(mouse_move_count),
                "mouse_click_count": int(mouse_click_count),
                "mouse_scroll_count": int(mouse_scroll_count),
                "mouse_total_distance_px": round(mouse_total_distance, 3),
                "mouse_mean_speed_px_s": round(mean_speed, 3),
                "mouse_max_speed_px_s": round(max_speed, 3),
                "active_seconds_fraction": round(active_frac, 6),
                # sequences for encoders (window only)
                "mouse_events": mouse_win,
                "key_events": keys_seq,
            }

    def prune_before(self, t_cutoff: float):
        """Drop events earlier than t_cutoff (seconds)."""
        with self.lock:
            t_cut_ms = float(t_cutoff) * 1000.0
            self.key_events = [ev for ev in self.key_events if float(ev.get("down_ts", 0.0)) >= t_cutoff]
            self.mouse_events = [ev for ev in self.mouse_events if float(ev[0]) >= t_cut_ms]
            self.active_seconds = {s for s in self.active_seconds if s >= int(t_cutoff)}

# ---------------- sequences → embeddings ----------------
def _z_norm(x: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray], eps=1e-6) -> np.ndarray:
    if mean is None or std is None: return x
    return (x - mean) / (std + eps)

def _load_norm_stats(path_npz: Path) -> Dict[str, np.ndarray]:
    stats = {}
    if path_npz.exists():
        d = np.load(path_npz, allow_pickle=True)
        for k in d.files:
            stats[k] = d[k]
    return stats

def _build_mouse_seq(mouse_events: List[Tuple[float, float, float, float]], stats: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if not mouse_events or len(mouse_events) < 2:
        return None
    ev = np.asarray(mouse_events, dtype=np.float32)  # [T, 4] -> t_ms, x, y, type
    t, x, y, typ = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
    dt = np.clip(np.diff(t, prepend=t[0]), 1e-3, None)
    dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0])
    speed = np.sqrt(dx**2 + dy**2) / dt
    accel = np.diff(speed, prepend=speed[0]) / dt
    type01 = np.clip(typ / 2.0, 0.0, 1.0)  # 0 move, 1 click, 2 scroll → [0..1]
    seq = np.stack([dx, dy, dt, speed, accel, type01], axis=1)  # [T, 6]
    m_mean, m_std = stats.get("mouse_mean", None), stats.get("mouse_std", None)
    if isinstance(m_mean, np.ndarray) and m_mean.ndim == 1 and m_mean.shape[0] == seq.shape[1]:
        seq = _z_norm(seq, m_mean, m_std)
    return seq[None, :, :]  # [1, T, 6]

def _build_key_seq(key_events: List[Dict[str, float]], stats: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if not key_events:
        return None
    dwell, flight, ikg = [], [], []
    for i, ev in enumerate(key_events):
        kd = float(ev.get("down_ts", 0.0) or 0.0)
        ku = float(ev.get("up_ts", kd) or kd)
        nd = float(ev.get("next_down_ts", ku) or ku)
        dwell.append(max(ku - kd, 0.0))           # seconds
        flight.append(max(nd - ku, 0.0))          # seconds
        prev_kd = float(key_events[i - 1].get("down_ts", kd) if i > 0 else kd)
        ikg.append(max(kd - prev_kd, 0.0))        # seconds
    seq = np.stack([np.array(dwell), np.array(flight), np.array(ikg)], axis=1)  # [T, 3]
    k_mean, k_std = stats.get("key_mean", None), stats.get("key_std", None)
    if isinstance(k_mean, np.ndarray) and k_mean.ndim == 1 and k_mean.shape[0] == seq.shape[1]:
        seq = _z_norm(seq, k_mean, k_std)
    return seq[None, :, :]  # [1, T, 3]

# ---------------- CSV schema ----------------
EMB_M_PREFIX = "mouse_emb_"
EMB_K_PREFIX = "key_emb_"
MVP_FIELDS = [
    "ks_event_count","ks_keydowns","ks_keyups","ks_unique_keys",
    "ks_mean_dwell_ms","ks_median_dwell_ms","ks_p95_dwell_ms",
    "ks_mean_ikg_ms","ks_median_ikg_ms","ks_p95_ikg_ms",
    "mouse_move_count","mouse_click_count","mouse_scroll_count",
    "mouse_total_distance_px","mouse_mean_speed_px_s","mouse_max_speed_px_s",
    "active_seconds_fraction"
]
STATIC_FIELDS = [
    "user_id","session_id","t0_unix","t1_unix",
    "stress_prob","confident","coverage","n_frames","n_face_frames",
    "pred_emotion","pred_confidence",
    "has_mouse_emb","has_keys_emb"
]

def ensure_csv(path: Path, header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

def append_row(path: Path, header: List[str], row: Dict[str, Any]):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)

# ---------------- main ----------------
def main():
    args = get_args()
    out_csv = Path(args.csv)
    artifacts = Path(args.artifacts_dir)
    # Use canonical artifact names you have in your folder
    enc_mouse_path = artifacts / "encoder_mouse(2).onnx"
    enc_key_path   = artifacts / "encoder_keystroke(2).onnx"
    norm_stats_npz = artifacts / "norm_stats(2).npz"

    # Load encoders (+ stats) if available
    enc_mouse = enc_key = None
    dim_m = dim_k = 64  # defaults if model hides dims
    stats = {}
    if ort is not None and enc_mouse_path.exists() and enc_key_path.exists():
        enc_mouse = _OnnxEncoder(enc_mouse_path, "mouse")
        enc_key   = _OnnxEncoder(enc_key_path, "keyboard")
        dim_m = int(enc_mouse.out_D or 64)
        dim_k = int(enc_key.out_D or 64)
        stats = _load_norm_stats(norm_stats_npz)
        print(f"[encoders] mouse_D={dim_m} key_D={dim_k} stats={'yes' if stats else 'no'}")
    else:
        print("[encoders] Missing ONNX encoders; embeddings will be zero vectors.")

    # Build CSV header with dynamic embedding columns
    header = STATIC_FIELDS + MVP_FIELDS + \
             [f"{EMB_M_PREFIX}{i}" for i in range(dim_m)] + \
             [f"{EMB_K_PREFIX}{i}" for i in range(dim_k)]
    ensure_csv(out_csv, header)
    print(f"[CSV] Writing to: {out_csv.resolve()}")

    # Load face model
    face_session = None; face_in_name = None; face_out_name = None; face_in_shape = None
    face_detector = None
    if ort is not None and os.path.exists(args.emotion_onnx):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        face_session = ort.InferenceSession(args.emotion_onnx, sess_options=so, providers=["CPUExecutionProvider"])
        face_in_name  = face_session.get_inputs()[0].name
        face_out_name = face_session.get_outputs()[0].name
        face_in_shape = face_session.get_inputs()[0].shape
        face_detector = cv2.CascadeClassifier(_CASCADE_PATH)
        if face_detector.empty():
            raise RuntimeError("Cannot load Haar cascade.")
        print(f"[face] onnx={args.emotion_onnx}")
        print(f"[face] input={face_in_name} shape={face_in_shape}  output={face_out_name}")
    else:
        print("[face] ONNX model not available; stress_prob will be 0.0")

    # camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # behavior listeners
    beh = BehaviorAggregator()
    listeners = []
    if keyboard and mouse:
        def _on_press(k):  beh.on_keydown(k, time.time())
        def _on_release(k): beh.on_keyup(k, time.time())
        kl = keyboard.Listener(on_press=_on_press, on_release=_on_release); kl.start(); listeners.append(kl)
        def _on_move(x, y): beh.on_move(x, y, time.time())
        def _on_click(x, y, b, pressed): 
            if pressed: beh.on_click(x, y, b, True, time.time())
        def _on_scroll(x, y, dx, dy): beh.on_scroll(x, y, dx, dy, time.time())
        ml = mouse.Listener(on_move=_on_move, on_click=_on_click, on_scroll=_on_scroll); ml.start(); listeners.append(ml)
        if args.poll_hz > 0:
            try:
                import pyautogui
                stop_poll = threading.Event()
                def _poll_mouse():
                    period = 1.0 / args.poll_hz
                    while not stop_poll.is_set():
                        x, y = pyautogui.position()
                        beh.on_move(x, y, time.time())
                        time.sleep(period)
                tp = threading.Thread(target=_poll_mouse, daemon=True); tp.start()
                listeners.append(("poll", stop_poll))
            except Exception:
                print("[INFO] pyautogui not available; skipping polling.")
    else:
        print("[INFO] Keyboard/mouse hooks not active; behavior features may be zero.")

    # windowing (overlap): emit every hop_sec, window length = win_sec
    win_sec = int(args.window_sec)
    hop_sec = int(args.hop_sec)
    min_face_cov = float(args.min_face_coverage)

    now = time.time()
    t_last_emit = (int(now) // hop_sec) * hop_sec  # align to hop boundary

    # per-frame cache for coverage/averaged stress in each window
    # items: (t_unix, used_face, stress_prob)
    frame_log = []

    fps_last = time.time()
    fps_counter = 0
    fps = 0.0

    inferred_layout = (True, 3, 224, 224)
    if face_session is not None and face_in_shape is not None:
        inferred_layout = _detect_layout(face_in_shape)

    print(f"[hybrid-logger] Running with {win_sec}s windows, {hop_sec}s hop (overlap ON) → writing to {args.csv}")
    session_id = args.session_id or time.strftime("%Y%m%d_%H%M")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to capture frame.")
                break
            frame = cv2.flip(frame, 1)
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # face detect & infer
            stress_prob = None
            pred_label_for_display = "—"
            pred_conf_for_display = 0.0
            if face_session is not None:
                faces = face_detector.detectMultiScale(gray_full, 1.1, 5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30,30))
                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
                    crop = frame[y:y+h, x:x+w]
                    enhanced = _enhance_gray(crop)
                    is_nchw, C, H, W = inferred_layout
                    logits = None
                    # a few fallback combos (layout, channels)
                    combos = [
                        (is_nchw, C),
                        (is_nchw, 1 if C == 3 else 3),
                        (not is_nchw, C),
                        (not is_nchw, 1 if C == 3 else 3),
                    ]
                    for layout, ch in combos:
                        x_in = _prep_tensor(enhanced, layout, ch, H, W)
                        try:
                            logits = face_session.run([face_out_name], {face_in_name: x_in})[0]
                            break
                        except Exception:
                            logits = None
                    if logits is not None:
                        vec = logits[0] if logits.ndim == 2 else np.squeeze(logits)
                        vec = vec[:len(CLASS_NAMES)]
                        probs = _softmax(vec)
                        top_idx = int(np.argmax(probs))
                        pred_label_for_display = CLASS_NAMES[top_idx]
                        pred_conf_for_display = float(probs[top_idx])
                        stress_prob = float(sum(probs[i] for i in sorted(ANXIETY_IDX)))
                        if args.display:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                            cv2.putText(frame, f"{pred_label_for_display}: {pred_conf_for_display*100:.1f}%", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # per-frame logging
            tnow = time.time()
            frame_log.append((tnow, stress_prob is not None, float(stress_prob) if stress_prob is not None else 0.0))

            fps_counter += 1
            if tnow - fps_last >= 1.0:
                fps = fps_counter / (tnow - fps_last)
                fps_counter = 0
                fps_last = tnow

            if args.display:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,230), 2)
                cv2.imshow("Hybrid Stress Logger", frame)

            # emit every hop_sec
            if (tnow - t_last_emit) >= hop_sec:
                t1 = int(tnow)
                t0 = t1 - win_sec

                # coverage + window-avg stress in [t0, t1)
                slice_frames = [(t, u, p) for (t, u, p) in frame_log if t0 <= t < t1]
                n_frames = len(slice_frames)
                n_face_frames = sum(1 for (_, u, _) in slice_frames if u)
                coverage = (n_face_frames / max(1, n_frames)) if n_frames > 0 else 0.0
                window_stress = float(np.mean([p for (_, u, p) in slice_frames if u])) if n_face_frames > 0 else 0.0
                confident = 1 if coverage >= min_face_cov else 0

                # Behavior summary JUST for this window
                beh_row = beh.summarize(t0, t1)

                # modality masks (help downstream head)
                has_mouse = 1 if beh_row.get("mouse_move_count", 0) >= 100 else 0
                has_keys  = 1 if beh_row.get("ks_keydowns", 0) >= 20 else 0

                # sequences → embeddings
                emb_m = np.zeros((1, dim_m), dtype=np.float32)
                emb_k = np.zeros((1, dim_k), dtype=np.float32)
                if enc_mouse is not None:
                    mseq = _build_mouse_seq(beh_row.get("mouse_events", []), stats)
                    if mseq is not None:
                        e = enc_mouse(mseq)
                        emb_m[:, :min(dim_m, e.shape[1])] = e[:, :min(dim_m, e.shape[1])]
                if enc_key is not None:
                    kseq = _build_key_seq(beh_row.get("key_events", []), stats)
                    if kseq is not None:
                        e = enc_key(kseq)
                        emb_k[:, :min(dim_k, e.shape[1])] = e[:, :min(dim_k, e.shape[1])]

                # build row
                row = {
                    # ---- static/meta ----
                    "user_id": args.user_id,
                    "session_id": session_id,
                    "t0_unix": round(float(t0), 3),
                    "t1_unix": round(float(t1), 3),
                    # ---- face supervision ----
                    "stress_prob": round(float(window_stress), 6),
                    "confident": int(confident),
                    "coverage": round(float(coverage), 6),
                    "n_frames": int(n_frames),
                    "n_face_frames": int(n_face_frames),
                    "pred_emotion": pred_label_for_display,
                    "pred_confidence": round(float(pred_conf_for_display), 6),
                    # ---- modality presence flags ----
                    "has_mouse_emb": int(has_mouse),
                    "has_keys_emb": int(has_keys),
                }

                # add 17 MVP
                for k in MVP_FIELDS:
                    v = beh_row.get(k, 0.0)
                    row[k] = float(v) if isinstance(v, (int, float)) else (int(v) if isinstance(v, bool) else 0.0)

                # add embeddings
                for i in range(dim_m):
                    row[f"{EMB_M_PREFIX}{i}"] = float(emb_m[0, i])
                for i in range(dim_k):
                    row[f"{EMB_K_PREFIX}{i}"] = float(emb_k[0, i])

                append_row(out_csv, header, row)
                try:
                    print("Logged:", {
                        "t": f"{time.strftime('%H:%M:%S', time.localtime(t1))}",
                        "stress": row["stress_prob"], "cov": row["coverage"],
                        "ks": row.get("ks_keydowns", 0), "mm": row.get("mouse_move_count", 0),
                        f"{EMB_M_PREFIX}0": row.get(f"{EMB_M_PREFIX}0", 0.0),
                        f"{EMB_K_PREFIX}0": row.get(f"{EMB_K_PREFIX}0", 0.0),
                    })
                except Exception:
                    pass

                # advance & prune (keep ≈2 windows of history)
                t_last_emit = t1
                beh.prune_before(t1 - (2 * win_sec))
                frame_log = [(t, u, p) for (t, u, p) in frame_log if t >= (t1 - 2 * win_sec)]

            if args.display and (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        for lst in listeners:
            if isinstance(lst, tuple) and lst[0] == "poll":
                lst[1].set()
            else:
                try: lst.stop()
                except: pass
        print("Closed.")

if __name__ == "__main__":
    main()
