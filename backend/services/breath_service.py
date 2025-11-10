# services/breath_service.py
# Runtime deps: only stdlib + Flask types
import time, uuid
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# ----------------------------
# Config (safe defaults)
# ----------------------------
DEFAULT_WINDOW_SEC = 60
EMA_ALPHA_DEFAULT = 0.2         # ~10-15s half-life for stress smoothing
HYST_MARGIN = 0.07              # to avoid mode flapping near band edges
RAMP_LIMIT_FRAC = 0.10          # max 10% cycle length change per plan
FREEZE_ON_UNSTABLE_SEC = 30     # keep current pattern if signal is unstable
STALE_TOLERANCE_SEC = 10        # if stress not updated recently, treat as stale

# Mode bands (based on smoothed stress)
# Calm:      [0.00, 0.25)
# Focus:     [0.25, 0.50)
# Wind-down: [0.50, 0.75)
# Relief:    [0.75, 1.00]
BANDS = [
    ("Calm",      0.00, 0.25, (4.0, 2.0, 4.0, 0.0)),  # inhale, hold, exhale, hold
    ("Focus",     0.25, 0.50, (4.0, 0.0, 6.0, 0.0)),  # longer exhale
    ("Wind-down", 0.50, 0.75, (4.0, 7.0, 8.0, 0.0)),  # 4-7-8
    ("Relief",    0.75, 1.01, (3.0, 0.0, 6.0, 0.0)),  # simple 1:2 under spikes
]

# Hard guardrails
MIN_MAX = {
    "inhale": (2.0, 6.0),
    "hold1":  (0.0, 7.0),
    "exhale": (4.0, 10.0),
    "hold2":  (0.0, 2.0),
}

AFFIRMATIONS = {
    "Calm":      "Slow, even breaths. Shoulders soft.",
    "Focus":     "Inhale ease; exhale tension. You’re doing fine.",
    "Wind-down": "Let the long exhale settle the body.",
    "Relief":    "Short inhale, longer exhale. You’re safe right now.",
    "Fallback":  "Breathe gently. Let the jaw unclench.",
}

# ----------------------------
# In-memory state
# ----------------------------
@dataclass
class UserBreathState:
    stress_raw: float = 0.0
    stress_smoothed: float = 0.0
    last_ts: float = 0.0
    last_mode: str = "Calm"
    last_cycle_len: float = sum(BANDS[0][3])  # default Calm cycle
    freeze_until: float = 0.0
    ema_alpha: float = EMA_ALPHA_DEFAULT
    # minimal session bookkeeping
    active_session_id: Optional[str] = None
    session_started_at: Optional[float] = None

_state: Dict[str, UserBreathState] = {}

def _now() -> float:
    return time.time()

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _pick_band_with_hysteresis(s: UserBreathState) -> Tuple[str, Tuple[float,float,float,float]]:
    """
    Use band edges with hysteresis margin to avoid flapping.
    """
    x = s.stress_smoothed
    current = s.last_mode
    chosen = None

    for name, lo, hi, pattern in BANDS:
        lo_h = lo + (HYST_MARGIN if name != current else 0.0)
        hi_h = hi - (HYST_MARGIN if name != current else 0.0)
        if x >= lo_h and x < hi_h:
            chosen = (name, pattern)
            break

    if chosen is None:
        # if nothing matched due to margins, fall back to current band by raw edges
        for name, lo, hi, pattern in BANDS:
            if x >= lo and x < hi:
                chosen = (name, pattern)
                break

    if chosen is None:
        chosen = ("Calm", BANDS[0][3])

    return chosen[0], chosen[1]

def _apply_ramp_limit(prev_len: float, new_len: float) -> float:
    if prev_len <= 0:
        return new_len
    delta = new_len - prev_len
    max_step = abs(prev_len) * RAMP_LIMIT_FRAC
    if abs(delta) <= max_step:
        return new_len
    return prev_len + (max_step if delta > 0 else -max_step)

def _pattern_guardrails(pat: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    i, h1, e, h2 = pat
    i = _clamp(i, *MIN_MAX["inhale"])
    h1 = _clamp(h1, *MIN_MAX["hold1"])
    e = _clamp(e, *MIN_MAX["exhale"])
    h2 = _clamp(h2, *MIN_MAX["hold2"])
    return (i, h1, e, h2)

def _cycles_for_window(cycle_len: float, window_sec: int) -> int:
    if cycle_len <= 0: return 1
    cycles = int(window_sec // cycle_len)
    return max(cycles, 1)

# ----------------------------
# Public API (service-level)
# ----------------------------
def init_service():
    """No-op initializer for symmetry with other services."""
    return True

def push_face_stress(user_id: Optional[str], stress_prob: float, face_present: bool = True, quality: str = "ok"):
    """
    Update stress cache from the face pipeline.
    """
    uid = str(user_id) if user_id is not None else "anon"
    s = _state.get(uid) or UserBreathState()
    ts = _now()

    # Treat missing face as unstable → freeze
    if not face_present or quality != "ok":
        s.freeze_until = max(s.freeze_until, ts + FREEZE_ON_UNSTABLE_SEC)
    else:
        # EMA smoothing
        if s.last_ts <= 0:
            s.stress_smoothed = float(stress_prob)
        else:
            a = s.ema_alpha
            s.stress_smoothed = (a * float(stress_prob)) + ((1.0 - a) * s.stress_smoothed)
        s.stress_raw = float(stress_prob)

    s.last_ts = ts
    _state[uid] = s

def get_status(user_id: Optional[str]) -> Dict[str, Any]:
    uid = str(user_id) if user_id is not None else "anon"
    s = _state.get(uid) or UserBreathState()
    stale = (_now() - s.last_ts) > STALE_TOLERANCE_SEC
    return {
        "user_id": uid,
        "stress_raw": s.stress_raw,
        "stress_smoothed": s.stress_smoothed,
        "last_mode": s.last_mode,
        "signal_quality": "stale" if stale else ("unstable" if _now() < s.freeze_until else "ok"),
        "last_update_age_sec": max(0.0, _now() - s.last_ts),
        "active_session_id": s.active_session_id,
    }

def plan(user_id: Optional[str], window_sec: int = DEFAULT_WINDOW_SEC) -> Dict[str, Any]:
    uid = str(user_id) if user_id is not None else "anon"
    s = _state.get(uid) or UserBreathState()
    ts = _now()

    # If stale or unstable → freeze or fallback to Calm
    unstable = ts < s.freeze_until
    stale = (ts - s.last_ts) > STALE_TOLERANCE_SEC

    if unstable:
        mode = s.last_mode
        # keep previous cycle_len if available; otherwise derive from current mode
        base_pat = next((p for (n, _, __, p) in [(n, lo, hi, p) for (n, lo, hi, p) in BANDS] if n == mode), BANDS[0][3])
    else:
        # pick a band with hysteresis
        mode, base_pat = _pick_band_with_hysteresis(s)
        s.last_mode = mode

    # Guardrails
    base_pat = _pattern_guardrails(base_pat)
    candidate_cycle_len = sum(base_pat)

    # Apply ramp limit vs previous cycle length (for smoothness)
    cycle_len = _apply_ramp_limit(s.last_cycle_len, candidate_cycle_len)
    s.last_cycle_len = cycle_len
    _state[uid] = s

    cycles = _cycles_for_window(cycle_len, window_sec)
    pattern = [
        {"phase": "inhale", "seconds": base_pat[0]},
        {"phase": "hold",   "seconds": base_pat[1]},
        {"phase": "exhale", "seconds": base_pat[2]},
        {"phase": "hold",   "seconds": base_pat[3]},
    ]

    note = "Using stable pace" if unstable or stale else "Adaptive pace from face stress"
    affirmation = AFFIRMATIONS.get(mode, AFFIRMATIONS["Fallback"])

    return {
        "start_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "mode": mode,
        "stress_smoothed": s.stress_smoothed,
        "cycles": cycles,
        "pattern": pattern,
        "notes": note,
        "affirmation": affirmation,
        "signal_quality": "stale" if stale else ("unstable" if unstable else "ok"),
        "window_sec": window_sec,
    }

def start_session(user_id: Optional[str], duration_target_sec: int = 180, with_audio: bool = False) -> Dict[str, Any]:
    uid = str(user_id) if user_id is not None else "anon"
    s = _state.get(uid) or UserBreathState()
    if s.active_session_id:
        return {"session_id": s.active_session_id, "started_at": s.session_started_at, "message": "already active"}
    s.active_session_id = str(uuid.uuid4())
    s.session_started_at = _now()
    _state[uid] = s
    return {"session_id": s.active_session_id, "started_at": s.session_started_at, "duration_target_sec": duration_target_sec, "with_audio": with_audio}

def stop_session(user_id: Optional[str]) -> Dict[str, Any]:
    uid = str(user_id) if user_id is not None else "anon"
    s = _state.get(uid) or UserBreathState()
    if not s.active_session_id:
        return {"message": "no active session"}
    sid = s.active_session_id
    started = s.session_started_at or _now()
    s.active_session_id = None
    s.session_started_at = None
    _state[uid] = s
    return {"session_id": sid, "duration_sec": max(0.0, _now() - started)}

def ingest_telemetry(user_id: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional adherence events (phase_completed, pause/resume etc.). For now, we just ack.
    """
    uid = str(user_id) if user_id is not None else "anon"
    # Could aggregate simple counters if desired.
    return {"ok": True, "user_id": uid, "ack": True}

# ----------------------------
# Helper used by app.py glue
# ----------------------------
def update_from_face_response(user_id: Optional[str], face_json: Dict[str, Any]):
    """
    Called after /api/stress/face/predict returns. Extracts stress_prob & updates cache.
    """
    try:
        faces = int(face_json.get("faces", 0))
    except Exception:
        faces = 0
    stress_prob = float(face_json.get("stress_prob", 0.0))
    face_present = faces > 0
    quality = "ok" if face_present else "no_face"
    push_face_stress(user_id=user_id, stress_prob=stress_prob, face_present=face_present, quality=quality)
