# services/stress_face_service.py
# Runtime deps: onnxruntime, opencv-python-headless, numpy, flask
import os, time, base64, traceback
from typing import Tuple, Optional, List, Any, Dict

import numpy as np
import cv2
import onnxruntime as ort
from flask import jsonify, Request

# ---------- Config ----------
_DEFAULT_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "export", "face_emotion_pt.onnx")
)
EMOTION_ONNX_PATH: str = os.getenv("EMOTION_ONNX_PATH", _DEFAULT_MODEL)

# MUST match training/test.py order
CLASS_NAMES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutrality", "sadness", "surprise"
]

# Anxiety mapping
NO_ANXIETY_IDX = {
    CLASS_NAMES.index("happiness"),
    CLASS_NAMES.index("contempt"),
    CLASS_NAMES.index("neutrality"),
}
ANXIETY_IDX = set(range(len(CLASS_NAMES))) - NO_ANXIETY_IDX

# Normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------- Globals ----------
_session: Optional[ort.InferenceSession] = None
_in_name: Optional[str] = None
_out_name: Optional[str] = None
_in_shape: Optional[List[Any]] = None
_detector: Optional[cv2.CascadeClassifier] = None

# ---------- Small utils ----------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(x) / x.size

def _detect_layout(shape: List[Any]) -> Tuple[bool, int, int, int]:
    """
    Return (is_nchw, C, H, W). Falls back if any dim is dynamic/None.
    """
    # Defaults
    is_nchw, C, H, W = True, 3, 224, 224
    if not isinstance(shape, list) or len(shape) != 4:
        return is_nchw, C, H, W

    n, d1, d2, d3 = shape  # [N,C,H,W] or [N,H,W,C]
    def _to_int(v, default):
        try:
            return int(v) if v is not None else default
        except Exception:
            return default

    d1i, d2i, d3i = _to_int(d1, None), _to_int(d2, None), _to_int(d3, None)

    # NCHW if channels at d1
    if d1i in (1, 3):
        return True, (d1i or C), (d2i or H), (d3i or W)
    # NHWC if channels at d3
    if d3i in (1, 3):
        return False, (d3i or C), (d1i or H), (d2i or W)

    return is_nchw, C, H, W

def _load_once():
    global _session, _in_name, _out_name, _in_shape, _detector
    if _session is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            EMOTION_ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"]
        )
        inp = _session.get_inputs()[0]
        out = _session.get_outputs()[0]
        _in_name, _out_name = inp.name, out.name
        _in_shape = inp.shape
        print(f"[face] onnx={EMOTION_ONNX_PATH}")
        print(f"[face] input={_in_name} shape={_in_shape}  output={_out_name} shape={out.shape}")

    if _detector is None:
        _detector = cv2.CascadeClassifier(_CASCADE_PATH)
        if _detector.empty():
            raise RuntimeError("Cannot load Haar cascade.")

def _read_image(req: Request) -> Optional[np.ndarray]:
    # multipart
    if req.files and "image" in req.files:
        data = np.frombuffer(req.files["image"].read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    # JSON {image: base64 or dataURL}
    j = req.get_json(silent=True) or {}
    v = j.get("image")
    if isinstance(v, str) and v:
        if v.startswith("data:"):
            v = v.split(",", 1)[-1]
        try:
            data = np.frombuffer(base64.b64decode(v, validate=True), np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None

def _largest_face(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _detector.detectMultiScale(gray, 1.1, 5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30,30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
    return bgr[y:y+h, x:x+w], (int(x), int(y), int(w), int(h))

# ---------- Preprocess builders (train–serve aligned) ----------
def _enhance_gray(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR -> gray -> CLAHE -> bilateral, returns enhanced grayscale."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return filtered

def _prep_tensor(enhanced_gray: np.ndarray, layout_nchw: bool, channels: int, h: int, w: int) -> np.ndarray:
    """
    Build input tensor per desired layout/channels.
    - channels == 1: grayscale, norm mean=0.5 std=0.5
    - channels == 3: convert gray->RGB, ImageNet normalize
    """
    if channels == 1:
        arr = cv2.resize(enhanced_gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        if layout_nchw:
            x = arr[None, None, :, :]      # (1,1,H,W)
        else:
            x = arr[:, :, None][None, ...] # (1,H,W,1)
        return x.astype(np.float32)

    # 3-channel
    rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    if layout_nchw:
        x = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1,3,H,W)
    else:
        x = rgb[None, ...]                            # (1,H,W,3)
    return x.astype(np.float32)

def _try_run(x: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Try a single run, return (logits, error_string)."""
    try:
        logits = _session.run([_out_name], {_in_name: x})[0]
        return logits, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _auto_infer(enhanced_gray: np.ndarray, inferred: Tuple[bool,int,int,int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Attempt model run with multiple input configurations until one succeeds.
    Order:
      1) inferred (layout, C, H, W)
      2) flip channels (3↔1) keep layout
      3) flip layout keep channels
      4) flip both
    Returns logits and a debug dict describing which combo worked (or last error).
    """
    is_nchw, C, H, W = inferred
    tried = []
    combos = [
        (is_nchw, C),
        (is_nchw, 1 if C == 3 else 3),
        (not is_nchw, C),
        (not is_nchw, 1 if C == 3 else 3),
    ]

    last_err = None
    for layout, ch in combos:
        x = _prep_tensor(enhanced_gray, layout, ch, H, W)
        logits, err = _try_run(x)
        tried.append({"layout": "NCHW" if layout else "NHWC", "channels": ch, "err": err})
        if err is None:
            return logits, {"picked_layout": "NCHW" if layout else "NHWC", "picked_channels": ch, "tried": tried}
        last_err = err

    # If all failed, raise a descriptive error
    raise RuntimeError(f"All input combinations failed. Last error: {last_err}", tried)

# ---------- Public API ----------
def face_health():
    try:
        _load_once()
        is_nchw, C, H, W = _detect_layout(_in_shape or [])
        ok = (
            _session is not None and _in_name is not None and _out_name is not None
            and _detector is not None and not _detector.empty()
            and os.path.exists(EMOTION_ONNX_PATH)
        )
        return jsonify({
            "ok": ok,
            "model_path": EMOTION_ONNX_PATH,
            "input_name": _in_name,
            "input_shape": _in_shape,
            "inferred_layout": "NCHW" if is_nchw else "NHWC",
            "inferred_channels": C,
            "class_names": CLASS_NAMES,
        }), 200 if ok else 500
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

def face_predict(req: Request):
    try:
        _load_once()
        bgr = _read_image(req)
        if bgr is None:
            return jsonify({"error": "No/invalid image"}), 400

        t0 = time.time()
        crop, bbox = _largest_face(bgr)
        if crop is None:
            return jsonify({"faces": 0, "elapsed_ms": int((time.time() - t0)*1000)}), 200

        inferred = _detect_layout(_in_shape or [])
        enhanced_gray = _enhance_gray(crop)

        # Auto-try input configs until one works
        logits, pick_info = _auto_infer(enhanced_gray, inferred)

        vec = logits[0] if logits.ndim == 2 else np.squeeze(logits)
        if vec.shape[-1] < len(CLASS_NAMES):
            return jsonify({
                "error": f"Model output size ({vec.shape[-1]}) < class count ({len(CLASS_NAMES)})",
                "model_out_shape": list(logits.shape),
                "pick_info": pick_info
            }), 500
        vec = vec[:len(CLASS_NAMES)]

        probs = _softmax(vec)
        top_idx = int(np.argmax(probs))
        top_emotion = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])
        anxiety_label = "No Anxiety" if top_idx in NO_ANXIETY_IDX else "Anxiety"
        stress_prob = float(sum(probs[i] for i in sorted(ANXIETY_IDX)))

        x0, y0, ww, hh = bbox
        return jsonify({
            "faces": 1,
            "bbox": [x0, y0, ww, hh],
            "label": anxiety_label,
            "confidence": top_conf,
            "top_emotion": top_emotion,
            "top_confidence": top_conf,
            "probs": probs.tolist(),
            "class_names": CLASS_NAMES,
            "stress_prob": stress_prob,
            "model_input": {
                "name": _in_name,
                "shape": _in_shape,
                "inferred_layout": "NCHW" if inferred[0] else "NHWC",
                "inferred_channels": inferred[1],
            },
            "pick_info": pick_info,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }), 200

    except Exception as e:
        # Return a helpful JSON error instead of a 500 with no body
        tb = traceback.format_exc(limit=2)
        return jsonify({
            "error": f"{type(e).__name__}: {e}",
            "trace": tb
        }), 500
