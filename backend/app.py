from flask import Flask, request, Response, stream_with_context, jsonify
from config import Config
from extensions import db
from flask_migrate import Migrate
from sqlalchemy import func

from services.auth_service import register_user, login_user
from services.tracking_service import start_tracking, stop_tracking
from services.stress_face_service import face_health, face_predict
from services.chatbot_service import chat_with_bot, reset_session, sse_stream
from services.stress_behavior_service import (
    init_service as behavior_init_service,
    health_check as behavior_health_check,
    predict_from_row as behavior_predict_from_row,
    latest_window_features as behavior_latest_window_features
)

from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from flask_cors import CORS

from models import Chat, ChatMessage, ChatSummary, UserMemory  # new models

import os
import signal
import atexit

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Initialize behavior predictor once at startup (CPU-only, lightweight)
behavior_init_service()

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


# -------- Auth --------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    return register_user(data)

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    return login_user(data)


# -------- Keyboard tracker--------
@app.route("/api/start_tracking", methods=["POST"])
def start_tracking_route():
    return start_tracking()

@app.route("/api/stop_tracking", methods=["POST"])
def stop_tracking_route():
    return stop_tracking()


# -------- Stress Face Detection  --------
@app.route("/api/stress/face/health", methods=["GET"])
@jwt_required(optional=True)
def stress_face_health():
    return face_health()

@app.route("/api/stress/face/predict", methods=["POST"])
@jwt_required(optional=True)
def stress_face_predict():
    uid = get_jwt_identity()
    return face_predict(request)  # returns (json, status)


# -------- Stress Behavior (keyboard/mouse) --------
@app.route("/api/stress/behavior/health", methods=["GET"])
@jwt_required(optional=True)
def stress_behavior_health():
    return jsonify(behavior_health_check())

@app.route("/api/stress/behavior/predict", methods=["POST"])
@jwt_required(optional=True)
def stress_behavior_predict():
    """
    POST body JSON (10s window recommended; 30s also works). Back-compat fields:
      - 17-feature MVP fields (same names as training)
      - Optional: has_mouse_emb, has_keys_emb (0/1)

    NEW (for embeddings via backend encoders):
      - "mouse_events": [[t_ms, x, y, type], ...]   # type: 0 move, 1 click, 2 scroll
        or "mouse_seq": [[dx, dy, dt, speed, accel, type01], ...]  # already derived
      - "key_events": [{"down_ts":..,"up_ts":..,"next_down_ts":..}, ...]
        or "key_seq": [[dwell_ms, flight_ms, ikg_ms], ...]

    Optional selector:
      - "head": "emb" | "hybrid"   # force a particular personal head; default = auto

    Optional:
      - "user_id": "harsh"
    """
    try:
        payload = request.get_json(silent=True) or {}
        uid = get_jwt_identity()
        if uid is not None and "user_id" not in payload:
            payload["user_id"] = str(uid)
        result = behavior_predict_from_row(payload, user_id=payload.get("user_id"))
        return jsonify({"ok": True, "result": result}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/stress/behavior/latest-window", methods=["GET"])
@jwt_required(optional=True)
def stress_behavior_latest_window():
    """
    Returns the latest aggregate MVP features from your labels CSV
    (prefers stress_hybrid_10s.csv; falls back to stress_30s.csv).
    """
    try:
        uid = get_jwt_identity()
        feats = behavior_latest_window_features(user_id=uid)
        return jsonify({"ok": True, "features": feats}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


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


if __name__ == "__main__":
    app.run(debug=True)
