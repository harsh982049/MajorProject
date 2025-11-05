import os, time, csv, argparse, threading
from pathlib import Path
import cv2, numpy as np
from tensorflow.keras.models import load_model

# Optional input listeners (for logging only; DO NOT affect stress)
try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = None
    mouse = None
    print("[WARN] pynput not installed. Keyboard/mouse fields will stay zero.")

# ---------------- CLI ----------------
def get_args():
    ap = argparse.ArgumentParser(description="Face-only stress (strict) + optional key/mouse logging → ONE CSV")
    ap.add_argument("--model", type=str, default="improved_emotion_recognition_model.h5")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--csv", type=str, default="stress_30s.csv")
    ap.add_argument("--window_sec", type=int, default=30)
    ap.add_argument("--min_face_coverage", type=float, default=0.30)
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--user_id", type=str, default="harsh")
    ap.add_argument("--session_id", type=str, default=None)
    ap.add_argument("--poll_hz", type=float, default=0.0)
    return ap.parse_args()

# ------------ Face helpers ------------
# EXACT same order as your working script:
EMOTION_LABELS = ['Anxiety', 'Anxiety', 'Anxiety', 'No Anxiety', 'Anxiety', 'Anxiety', 'No Anxiety']
NO_ANX_PREFIX = "no"  # detect “No Anxiety” robustly

def frame_to_face_tensor(gray, box, size=(48,48)):
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0: return None
    roi = cv2.resize(roi, size).astype(np.float32) / 255.0
    roi = np.expand_dims(roi, axis=(0, -1))  # [1,H,W,1]
    return roi

def norm_probs(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    s = float(np.sum(v))
    return v / s if s > 0 else np.ones_like(v) / len(v)

# ---------- Behavior aggregator (logging only) ----------
class BehaviorAggregator:
    def __init__(self):
        self.lock = threading.Lock()
        self.keydown_times = {}; self.dwell_times = []; self.ikg_times = []
        self.ks_event_count = 0; self.ks_keydowns = 0; self.ks_keyups = 0; self.ks_keys_seen = set(); self.last_keydown_time=None
        self.mouse_move_count = 0; self.mouse_click_count = 0; self.mouse_scroll_count = 0
        self.mouse_total_distance = 0.0; self.mouse_last_pos=None; self.mouse_last_time=None; self.mouse_speeds=[]
        self.active_seconds = set()
    def reset(self): 
        with self.lock: self.__init__()
    def _mark(self,t): self.active_seconds.add(int(t))
    # keyboard
    def on_keydown(self,k,t):
        with self.lock:
            self.ks_event_count += 1; self.ks_keydowns += 1; self.ks_keys_seen.add(str(k))
            if self.last_keydown_time is not None: self.ikg_times.append((t-self.last_keydown_time)*1000.0)
            self.last_keydown_time = t; self.keydown_times[str(k)] = t; self._mark(t)
    def on_keyup(self,k,t):
        with self.lock:
            self.ks_event_count +=1; self.ks_keyups +=1; kk=str(k)
            if kk in self.keydown_times:
                dt=(t-self.keydown_times[kk])*1000.0
                if 0<=dt<5000: self.dwell_times.append(dt)
                del self.keydown_times[kk]
            self._mark(t)
    # mouse
    def on_move(self,x,y,t):
        with self.lock:
            self.mouse_move_count += 1
            if self.mouse_last_pos is not None and self.mouse_last_time is not None:
                dx=x-self.mouse_last_pos[0]; dy=y-self.mouse_last_pos[1]; dist=(dx*dx+dy*dy)**0.5
                dt=max(1e-3,t-self.mouse_last_time); self.mouse_total_distance += dist; self.mouse_speeds.append(dist/dt)
            self.mouse_last_pos=(x,y); self.mouse_last_time=t; self._mark(t)
    def on_click(self,x,y,button,pressed,t):
        with self.lock:
            if pressed: self.mouse_click_count += 1; self._mark(t)
    def on_scroll(self,x,y,dx,dy,t):
        with self.lock:
            self.mouse_scroll_count += 1; self._mark(t)
    def summarize(self,t0,t1):
        with self.lock:
            def stats(a):
                if not a: return (0.0,0.0,0.0)
                a=np.array(a,dtype=np.float32); return float(np.mean(a)), float(np.median(a)), float(np.percentile(a,95))
            mdw,mdw_med,mdw_p95 = stats(self.dwell_times)
            mikg,mikg_med,mikg_p95 = stats(self.ikg_times)
            mean_speed = float(np.mean(self.mouse_speeds)) if self.mouse_speeds else 0.0
            max_speed  = float(np.max(self.mouse_speeds)) if self.mouse_speeds else 0.0
            active_frac = len([s for s in self.active_seconds if (t0 <= s < t1)]) / max(1.0,(t1-t0))
            return {
                "ks_event_count": int(self.ks_event_count),
                "ks_keydowns": int(self.ks_keydowns),
                "ks_keyups": int(self.ks_keyups),
                "ks_unique_keys": int(len(self.ks_keys_seen)),
                "ks_mean_dwell_ms": round(mdw,3),
                "ks_median_dwell_ms": round(mdw_med,3),
                "ks_p95_dwell_ms": round(mdw_p95,3),
                "ks_mean_ikg_ms": round(mikg,3),
                "ks_median_ikg_ms": round(mikg_med,3),
                "ks_p95_ikg_ms": round(mikg_p95,3),
                "mouse_move_count": int(self.mouse_move_count),
                "mouse_click_count": int(self.mouse_click_count),
                "mouse_scroll_count": int(self.mouse_scroll_count),
                "mouse_total_distance_px": round(self.mouse_total_distance,3),
                "mouse_mean_speed_px_s": round(mean_speed,3),
                "mouse_max_speed_px_s": round(max_speed,3),
                "active_seconds_fraction": round(active_frac,6),
            }

# -------------- CSV --------------
CSV_FIELDS = [
    "user_id","session_id","t0_unix","t1_unix",
    # face (strict face-only)
    "stress_prob","confident","coverage","n_frames","n_face_frames",
    "pred_emotion","pred_confidence",
    # keyboard (logging only)
    "ks_event_count","ks_keydowns","ks_keyups","ks_unique_keys",
    "ks_mean_dwell_ms","ks_median_dwell_ms","ks_p95_dwell_ms",
    "ks_mean_ikg_ms","ks_median_ikg_ms","ks_p95_ikg_ms",
    # mouse (logging only)
    "mouse_move_count","mouse_click_count","mouse_scroll_count",
    "mouse_total_distance_px","mouse_mean_speed_px_s","mouse_max_speed_px_s",
    "active_seconds_fraction"
]
def ensure_csv(p, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p,"w",newline="") as f: csv.DictWriter(f, fieldnames=fields).writeheader()
def append_row(p, fields, row):
    with open(p,"a",newline="") as f: csv.DictWriter(f, fieldnames=fields).writerow(row)

# -------------- Main --------------
def main():
    args = get_args()
    out_csv = Path(args.csv); ensure_csv(out_csv, CSV_FIELDS)

    user_id = args.user_id
    session_id = args.session_id or time.strftime("%Y%m%d_%H%M")
    win_sec = int(args.window_sec)
    min_cov = float(args.min_face_coverage)
    poll_hz = float(args.poll_hz)

    # Load model
    print(f"Loading face model: {args.model}")
    model = load_model(args.model); print("Model loaded.")

    # Camera + detector
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened(): print("Error: Could not open webcam"); return
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Behavior listeners (logging only)
    beh = BehaviorAggregator()
    listeners=[]
    if keyboard and mouse:
        def _on_press(k):  beh.on_keydown(k, time.time())
        def _on_release(k): beh.on_keyup(k, time.time())
        kl = keyboard.Listener(on_press=_on_press, on_release=_on_release); kl.start(); listeners.append(kl)
        def _on_move(x,y): beh.on_move(x,y,time.time())
        def _on_click(x,y,b,pressed): 
            if pressed: beh.on_click(x,y,b,True,time.time())
        def _on_scroll(x,y,dx,dy): beh.on_scroll(x,y,dx,dy,time.time())
        ml = mouse.Listener(on_move=_on_move, on_click=_on_click, on_scroll=_on_scroll); ml.start(); listeners.append(ml)
        if poll_hz>0:
            try:
                import pyautogui
                stop_poll = threading.Event()
                def _poll_mouse():
                    period=1.0/poll_hz
                    while not stop_poll.is_set():
                        x,y=pyautogui.position(); beh.on_move(x,y,time.time()); time.sleep(period)
                tp = threading.Thread(target=_poll_mouse, daemon=True); tp.start()
                listeners.append(("poll", stop_poll))
            except Exception:
                print("[INFO] pyautogui not available; skipping polling.")
    else:
        print("[INFO] Keyboard/mouse hooks not active; only face fields will populate.")

    # Align window to grid
    now=time.time()
    t_window_start=(int(now)//win_sec)*win_sec
    if t_window_start+win_sec<=now: t_window_start+=win_sec

    frame_count=0; face_frame_count=0
    # strict face-only stress accumulation
    stress_sum = 0.0  # add 'confidence' only when predicted class is Anxiety
    pred_label_for_display="—"; pred_conf_for_display=0.0

    fps_last=time.time(); fps_counter=0; fps=0.0
    print(f"[CSV] Writing to: {out_csv.resolve()}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: print("Error: Failed to capture frame."); break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces)>0:
                # largest face
                areas=[w*h for (x,y,w,h) in faces]; i=int(np.argmax(areas))
                x,y,w,h = faces[i]
                face_t = frame_to_face_tensor(gray, (x,y,w,h))
                if face_t is not None:
                    raw = model.predict(face_t, verbose=0)[0]
                    probs = norm_probs(raw)
                    idx = int(np.argmax(probs))
                    label = EMOTION_LABELS[idx]
                    conf  = float(probs[idx])

                    # strict face-only accumulation:
                    if not label.lower().startswith(NO_ANX_PREFIX):  # any Anxiety class
                        stress_sum += conf   # add ONLY confidence of Anxiety class
                    # else add 0

                    face_frame_count += 1
                    pred_label_for_display = label
                    pred_conf_for_display  = conf

                    # draw like your original
                    if args.display:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(frame, f"{label}: {conf*100:.1f}%", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            frame_count += 1
            fps_counter += 1
            tnow=time.time()
            if tnow - fps_last >= 1.0:
                fps = fps_counter/(tnow-fps_last); fps_counter=0; fps_last=tnow
            if args.display:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,230), 2)
                cv2.imshow("Face-only Stress + (optional) Behavior Logger", frame)

            # window rollover
            if (tnow - t_window_start) >= win_sec:
                t0=t_window_start; t1=t_window_start+win_sec
                coverage = face_frame_count/max(1,frame_count)
                confident = 1 if coverage >= min_cov else 0

                # strict face-only window stress:
                # mean over ALL face frames of (conf if Anxiety, else 0)
                stress_prob = (stress_sum / face_frame_count) if face_frame_count>0 else 0.0

                # behavior summary (logged only; not used in stress)
                beh_row = BehaviorAggregator().summarize(0,1)  # fallback zeros
                if keyboard and mouse:
                    beh_row = beh.summarize(t0,t1)

                row = {
                    "user_id": user_id, "session_id": session_id,
                    "t0_unix": round(t0,3), "t1_unix": round(t1,3),
                    "stress_prob": round(float(stress_prob), 6),  # 0..1
                    "confident": int(confident),
                    "coverage": round(float(coverage), 6),
                    "n_frames": int(frame_count), "n_face_frames": int(face_frame_count),
                    "pred_emotion": pred_label_for_display,
                    "pred_confidence": round(float(pred_conf_for_display), 6),
                    **beh_row
                }
                append_row(out_csv, CSV_FIELDS, row)
                print("Logged:", row)

                # reset
                t_window_start=t1
                frame_count=0; face_frame_count=0; stress_sum=0.0
                pred_label_for_display="—"; pred_conf_for_display=0.0
                if keyboard and mouse: beh.reset()

            if args.display and (cv2.waitKey(1) & 0xFF) == ord('q'): break

    finally:
        cap.release()
        if args.display: cv2.destroyAllWindows()
        for lst in listeners:
            if isinstance(lst, tuple) and lst[0]=="poll":
                lst[1].set()
            else:
                try: lst.stop()
                except: pass
        print("Closed.")

if __name__ == "__main__":
    main()
