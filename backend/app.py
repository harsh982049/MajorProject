# app.py (updated)
from flask import Flask, request, Response, stream_with_context, jsonify, make_response
from config import Config
from extensions import db
from flask_migrate import Migrate
from sqlalchemy import func

from services.auth_service import register_user, login_user
from services.stress_face_service import face_health, face_predict
from services.chatbot_service import chat_with_bot, reset_session, sse_stream

# NEW: Breathing coach service
from services.breath_service import (
    init_service as breath_init_service,
    get_status as breath_get_status,
    plan as breath_plan,
    start_session as breath_start_session,
    stop_session as breath_stop_session,
    ingest_telemetry as breath_ingest_telemetry,
    update_from_face_response as breath_update_from_face_response,
)

# NEW: Focus Companion (planner + scheduler + notifier) services
from services.supabase_client import supabase  # Supabase server client
from services.focus_planner_service import plan_subtasks
from services.scheduler_service import schedule_subtasks
from services.notifier_service import render_email_for_subtask, send_email
from services.hooks_service import handle_magic_hook

from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from flask_cors import CORS

from models import Chat, ChatMessage, ChatSummary, UserMemory  # existing models

import os
import signal
import atexit
from datetime import datetime, timezone, timedelta
import pytz
from postgrest.exceptions import APIError

# Scheduler (for email reminders)
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Initialize services on startup
breath_init_service()

# -----------------------------------------
# PID-based tracker cleanup on Flask exit
# -----------------------------------------
def kill_tracker():
    pid_file = "tracker_tray.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"✅ Tracker process (PID {pid}) terminated on Flask exit.")
            os.remove(pid_file)
        except Exception as e:
            print(f"⚠️ Error terminating tracker: {e}")

atexit.register(kill_tracker)
signal.signal(signal.SIGINT, lambda sig, frame: exit(0))  # handle Ctrl+C

# =========================================
# APScheduler for Focus Companion reminders
# =========================================
DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000")

scheduler = BackgroundScheduler(timezone=DEFAULT_TZ)
scheduler.start()

def _enqueue_email_job(subtask: dict, to_email: str):
    when = subtask.get("planned_start_ts")
    if not when:
        return
    if isinstance(when, str):
        when = datetime.fromisoformat(when.replace("Z",""))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)   # aware
    now = datetime.now(timezone.utc)               # aware
    job_id = f"email_{subtask['id']}"

    def _send_now():
        subject, html = render_email_for_subtask(subtask, PUBLIC_BASE_URL, to_email)
        send_email(to_email, subject, html)

        scheduled_iso = (
            subtask["planned_start_ts"].isoformat()
            if hasattr(subtask.get("planned_start_ts"), "isoformat")
            else str(subtask.get("planned_start_ts"))
        )

        supabase.table("notifications").insert({
            "subtask_id": subtask["id"],
            "channel": "email",
            "scheduled_ts": scheduled_iso,                               # <-- string
            "sent_ts": datetime.now(timezone.utc).isoformat(),           # <-- string
            "status": "sent"
        }).execute()

    try:
        scheduler.remove_job(job_id)
    except Exception:
        pass

    if when <= now:
        scheduler.add_job(_send_now, "date", run_date=now + timedelta(seconds=5), id=job_id)
    else:
        scheduler.add_job(_send_now, "date", run_date=when, id=job_id)


# -------- Auth --------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    return register_user(data)

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    return login_user(data)

# -------- Stress Face Detection --------
@app.route("/api/stress/face/health", methods=["GET"])
@jwt_required(optional=True)
def stress_face_health():
    return face_health()

@app.route("/api/stress/face/predict", methods=["POST"])
@jwt_required(optional=True)
def stress_face_predict():
    """
    Pass-through to face predictor, with an added hook:
    - After getting the JSON, we update the breathing service's stress cache so
      the breathing plan can adapt in (near) real-time without any frontend changes.
    """
    uid = get_jwt_identity()
    resp, status = face_predict(request)  # returns (Response, status_code)
    try:
        # Update breathing cache using the same user id
        payload = resp.get_json(silent=True) or {}
        breath_update_from_face_response(user_id=uid, face_json=payload)
    except Exception:
        # If anything goes wrong here, we still return the face response unchanged.
        pass
    return resp, status

# -------- Breathing Coach (Existing) --------
@app.route("/api/breath/status", methods=["GET"])
@jwt_required(optional=True)
def breath_status_route():
    uid = get_jwt_identity()
    return jsonify(breath_get_status(uid)), 200

@app.route("/api/breath/plan", methods=["GET"])
@jwt_required(optional=True)
def breath_plan_route():
    uid = get_jwt_identity()
    window = int(request.args.get("window", 60))
    return jsonify(breath_plan(uid, window_sec=window)), 200

@app.route("/api/breath/session", methods=["POST"])
@jwt_required(optional=True)
def breath_session_route():
    uid = get_jwt_identity()
    body = request.get_json(silent=True) or {}
    action = (body.get("action") or "start").lower()
    duration = int(body.get("duration_target_sec", 180))
    with_audio = bool(body.get("with_audio", False))
    if action == "stop":
        return jsonify(breath_stop_session(uid)), 200
    else:
        return jsonify(breath_start_session(uid, duration_target_sec=duration, with_audio=with_audio)), 200

@app.route("/api/breath/telemetry", methods=["POST"])
@jwt_required(optional=True)
def breath_telemetry_route():
    uid = get_jwt_identity()
    payload = request.get_json(silent=True) or {}
    return jsonify(breath_ingest_telemetry(uid, payload)), 200

# =========================================
# Focus Companion (Planner + Scheduler + Notifier)
# =========================================

def sb_select_one(table: str, **equals):
    """Return first row or None."""
    q = supabase.table(table).select("*")
    for k, v in equals.items():
        q = q.eq(k, v)
    res = q.limit(1).execute()
    rows = (res.data or []) if res else []
    return rows[0] if rows else None

def sb_upsert_one(table: str, payload: dict, on_conflict: str | None = None):
    """Upsert a single row and return the resulting row (first). Works with supabase-py v2."""
    if on_conflict:
        res = supabase.table(table).upsert(payload, on_conflict=on_conflict).execute()
    else:
        res = supabase.table(table).upsert(payload).execute()
    rows = (res.data or []) if res else []
    return rows[0] if rows else None


def get_or_create_user_by_email(email: str):
    user = sb_select_one("users", email=email)
    if user:
        return user
    # insert (or upsert by unique email)
    name = email.split("@")[0]
    return sb_upsert_one("users", {"email": email, "name": name}, on_conflict="email")

def ensure_prefs_for_user(user_id: str, defaults: dict):
    prefs = sb_select_one("user_prefs", user_id=user_id)
    if prefs:
        return prefs
    payload = {
        "user_id": user_id,
        "timezone": os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata"),
        "work_start_hhmm": defaults.get("work_start_hhmm", "06:00"),
        "work_end_hhmm": defaults.get("work_end_hhmm", "23:59"),
        "default_buffer_min": defaults.get("default_buffer_min", 1),
        "notify_email": defaults.get("notify_email", True),
        "notify_telegram": defaults.get("notify_telegram", False),
        "telegram_chat_id": defaults.get("telegram_chat_id"),
    }
    return sb_upsert_one("user_prefs", payload, on_conflict="user_id")


@app.post("/api/focus/task/create")
@jwt_required(optional=True)
def focus_create_task():
    data = request.get_json(force=True)

    user_email = (data.get("user_email") or "").strip()
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    description = (data.get("description") or None)
    timebox_min = int(data.get("timebox_min", 60))
    deadline_ts = data.get("deadline_ts")  # ISO or None
    constraints = data.get("constraints", {})

    # 1) Get-or-create user
    try:
        user = get_or_create_user_by_email(user_email)
        if not user:
            return jsonify({"error": "Failed to create/find user"}), 500
    except APIError as e:
        return jsonify({"error": f"users upsert/select failed: {e.message}"}), 500

    # 2) Ensure prefs exist (and are demo-friendly for quick tests)
    try:
        prefs = ensure_prefs_for_user(user["id"], {
            "work_start_hhmm": "06:00",
            "work_end_hhmm": "23:59",
            "default_buffer_min": 1,
            "notify_email": True
        })
        if not prefs:
            return jsonify({"error": "Failed to create/find user_prefs"}), 500
    except APIError as e:
        return jsonify({"error": f"user_prefs upsert/select failed: {e.message}"}), 500

    # 3) Insert task
    try:
        task_res = supabase.table("tasks").insert({
            "user_id": user["id"],
            "title": title,
            "description": description,
            "deadline_ts": deadline_ts,
            "timebox_min": timebox_min
        }).execute()
        task_rows = task_res.data or []
        if not task_rows:
            return jsonify({"error": "tasks insert returned no rows"}), 500
        task = task_rows[0]
    except APIError as e:
        return jsonify({"error": f"tasks insert failed: {e.message}"}), 500

    # 4) Plan with Gemini
    try:
        plan = plan_subtasks(title, timebox_min, constraints)
    except Exception as e:
        return jsonify({"error": f"planner failed: {str(e)}"}), 500

    steps_payload = [{
        "task_id": task["id"],
        "idx": s.idx,
        "title": s.title,
        "dod_text": s.definition_of_done,
        "estimate_min": s.estimate_min,
        "state": "scheduled"
    } for s in plan.steps]

    try:
        ins_res = supabase.table("subtasks").insert(steps_payload).execute()
        inserted = ins_res.data or []
        if not inserted:
            return jsonify({"error": "subtasks insert returned no rows"}), 500
    except APIError as e:
        return jsonify({"error": f"subtasks insert failed: {e.message}"}), 500

    # 5) Schedule
    now_utc = datetime.now(timezone.utc)
    scheduled = schedule_subtasks(
        now_utc=now_utc,
        tz_name=prefs["timezone"],
        work_start_hhmm=prefs["work_start_hhmm"],
        work_end_hhmm=prefs["work_end_hhmm"],
        buffer_min=prefs["default_buffer_min"],
        subtasks=inserted
    )
    try:
        for st in scheduled:
            supabase.table("subtasks").update({
                "planned_start_ts": st["planned_start_ts"].isoformat(),
                "planned_end_ts": st["planned_end_ts"].isoformat()
            }).eq("id", st["id"]).execute()
    except APIError as e:
        return jsonify({"error": f"subtasks update failed: {e.message}"}), 500

    # 6) Enqueue email reminders (non-fatal if it fails)
    try:
        if prefs.get("notify_email", True) and user.get("email"):
            for st in scheduled:
                _enqueue_email_job(st, user["email"])
    except Exception:
        pass

    return jsonify({
        "task_id": task["id"],
        "subtasks": [{
            "id": st["id"], "idx": st["idx"], "title": st["title"],
            "planned_start_ts": st["planned_start_ts"].isoformat(),
            "planned_end_ts": st["planned_end_ts"].isoformat()
        } for st in scheduled]
    }), 200

@app.get("/api/focus/hook")
def focus_magic_hook():
    """
    Magic-link endpoint for Start/Done/Snooze/Blocked actions from email/Telegram.
    Example: /api/focus/hook?token=...&action=done
    """
    token = request.args.get("token")
    action = request.args.get("action")
    ok, msg = handle_magic_hook(token, action, dict(request.args))
    html = f"<html><body style='font-family:Inter,Arial'><h3>{'✅' if ok else '⚠️'} {msg}</h3></body></html>"
    status = 200 if ok else 400
    resp = make_response(html, status)
    resp.headers["Content-Type"] = "text/html"
    return resp


# -------- Chat management --------
@app.route("/api/chats", methods=["POST"])
@jwt_required()  # must be logged in to create a persistent chat
def create_chat():
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        user_id = None

    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip() or None
    is_journal = bool(body.get("is_journal", False))

    if is_journal:
        existing = Chat.query.filter_by(user_id=user_id, is_journal=True).first()
        if existing:
            return jsonify({"chat_id": existing.id, "title": existing.title, "is_journal": True}), 200

    if not is_journal and title is None:
        candidates = (
            Chat.query
            .filter_by(user_id=user_id, is_journal=False)
            .order_by(
                func.coalesce(Chat.updated_at, Chat.created_at).desc(),
                Chat.id.desc()
            )
            .all()
        )
        for c in candidates:
            msg_count = ChatMessage.query.filter_by(chat_id=c.id).count()
            if msg_count == 0:
                return jsonify({"chat_id": c.id, "title": c.title, "is_journal": c.is_journal}), 200

    chat = Chat(user_id=user_id, title=title, is_journal=is_journal)
    db.session.add(chat)
    db.session.commit()
    return jsonify({"chat_id": chat.id, "title": chat.title, "is_journal": chat.is_journal}), 201

@app.route("/api/chats", methods=["GET"])
@jwt_required()
def list_chats():
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        return jsonify({"error": "Invalid token"}), 401
    rows = (
        Chat.query
        .filter_by(user_id=user_id)
        .order_by(
            func.coalesce(Chat.updated_at, Chat.created_at).desc(),
            Chat.id.desc()
        )
        .all()
    )
    out = []
    for c in rows:
        out.append({
            "chat_id": c.id,
            "title": c.title or "New conversation",
            "is_journal": c.is_journal,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        })
    return jsonify({"chats": out}), 200

@app.route("/api/chats/<int:chat_id>/messages", methods=["GET"])
@jwt_required()
def get_chat_messages(chat_id: int):
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        user_id = None
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    limit = int(request.args.get("limit", 100))
    rows = (ChatMessage.query
            .filter_by(chat_id=chat_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
            .all())
    rows = list(reversed(rows))
    return jsonify({"messages": [{"id": r.id, "role": r.role, "content": r.content, "created_at": r.created_at.isoformat()} for r in rows]})

@app.route("/api/chats/<int:chat_id>", methods=["DELETE"])
@jwt_required()
def delete_chat(chat_id: int):
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        user_id = None
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    ChatMessage.query.filter_by(chat_id=chat_id).delete()
    ChatSummary.query.filter_by(chat_id=chat_id).delete()
    db.session.delete(chat)
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200

# -------- User profile memory --------
@app.route("/api/profile/memory", methods=["GET"])
@jwt_required()
def get_profile_memory():
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        user_id = None
    rows = (UserMemory.query
            .filter_by(user_id=user_id)
            .order_by(UserMemory.score.desc(), UserMemory.updated_at.desc())
            .all())
    return jsonify({"items": [{"id": r.id, "key": r.key, "value": r.value, "score": r.score} for r in rows]})

@app.route("/api/profile/memory", methods=["POST"])
@jwt_required()
def upsert_profile_memory():
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        user_id = None
    body = request.get_json(silent=True) or {}
    items = body.get("items", [])
    for it in items:
        key = (it.get("key") or "").strip()
        val = (it.get("value") or "").strip()
        score = float(it.get("score", 0.5))
        if not key or not val:
            continue
        existing = UserMemory.query.filter_by(user_id=user_id, key=key).first()
        if existing:
            existing.value = val
            existing.score = score
        else:
            db.session.add(UserMemory(user_id=user_id, key=key, value=val, score=score))
    db.session.commit()
    return jsonify({"message": "Saved"}), 200

# -------- Chatbot (JSON & SSE) --------
@app.route("/api/chatbot", methods=["POST"])
@jwt_required(optional=True)
def chatbot_route():
    data = request.get_json()
    return chat_with_bot(data)  # returns (json, status)

@app.route("/api/chatbot/stream", methods=["GET"])
@jwt_required(optional=True)
def chatbot_stream_route():
    uid = get_jwt_identity()
    try:
        user_id = int(uid) if uid is not None else None
    except ValueError:
        user_id = None
    session_id = request.args.get("session_id", "")
    chat_id = request.args.get("chat_id", "")
    chat_id = int(chat_id) if (chat_id and chat_id.isdigit()) else None
    user_message = request.args.get("message", "")
    return Response(stream_with_context(
        sse_stream(user_id, chat_id, session_id, user_message)
    ), headers={
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })

@app.route("/api/chatbot/reset", methods=["POST"])
def chatbot_reset_route():
    data = request.get_json()
    return reset_session(data)

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    app.run(debug=True)
