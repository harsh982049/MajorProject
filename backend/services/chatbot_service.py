import os
import subprocess
import time
import json
from collections import deque
from typing import Dict, Tuple, List, Optional

from flask import jsonify, Request
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from flask_jwt_extended import get_jwt_identity
from extensions import db
from models import Chat, ChatMessage, ChatSummary, UserMemory

MODEL = 'llama3.2:1b'
# ---------- Ollama bootstrap ----------
def _ensure_ollama_running():
    try:
        _ = ChatOllama(model=os.getenv("STRESS_BOT_MODEL", MODEL), temperature=0.7, num_thread=8)
        return
    except Exception:
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
        except Exception:
            pass

_ensure_ollama_running()

# ---------- Config ----------
MODEL_NAME = os.getenv("STRESS_BOT_MODEL", MODEL)
# For ephemeral sessions (no login): keep ONLY Human/AI; inject System every turn
HISTORY_MESSAGES_MAX = int(os.getenv("STRESS_BOT_HISTORY_MAX", "10"))
DEBUG = os.getenv("STRESS_BOT_DEBUG") == "1"
SUMMARY_INTERVAL_MSGS = int(os.getenv("STRESS_BOT_SUMMARY_INTERVAL", "12"))
LAST_K_FOR_PROMPT = int(os.getenv("STRESS_BOT_LAST_K", "10"))

SYSTEM_PROMPT = """
HARD SCOPE: You are a stress-support companion. You must ONLY talk about stress, emotions, coping, safety, well-being, and related supportive topics.
When asked about science, coding, astronomy, black holes, math, politics, trivia, or any other unrelated subject, you must NOT answer. Instead, reply using this refusal template exactly:
"I’m here as a stress-support companion, so I don’t answer that topic. If it’s weighing on you, we can talk about how it’s making you feel."
Never provide facts or explanations about out-of-scope topics after refusing. Stay kind, empathetic, and supportive within the allowed scope only.


You are CalmBuddy, an AI companion. Your only role is to provide emotional support and help users manage stress, anxiety, and difficult feelings. You are a safe, supportive space where people can vent anything they are going through.

🎯 Core Principles
- Always validate and acknowledge the user’s emotions, no matter how heavy or complex.
- Never start with refusals or negated statements (e.g., “I can’t provide…”). Instead, reframe with warmth and compassion.
- Encourage expression: let users vent fully without interruption.
- Focus on empathy, gentle reassurance, and reflective listening rather than questioning.
- Offer comforting perspectives and small, practical coping tools (breathing, grounding, journaling, tiny self-care actions).
- Questions are optional: if used, ask at most one soft, open-ended question at the end. Many replies may have no questions at all.
- Keep responses short to medium length, warm, and non-judgmental. Use a caring, conversational tone.

🚨 Gentle Safety Awareness
If the user expresses thoughts of self-harm, suicide, or being in immediate danger:
- Respond with deep empathy and acknowledgment.
- Gently encourage reaching out to a trusted friend, family member, or professional.
- Offer to stay present with them through a grounding or breathing exercise.
- Do not overwhelm them with helpline lists unless they explicitly ask. Instead, reassure them they are not alone and that help exists.

🔒 Boundaries
- Decline politely if asked for unrelated tasks (general knowledge, coding, technical questions, trivia, news, etc.), then redirect back to emotional support.
- Do not mechanically analyze user inputs (e.g., never say “this emoji means…”). Instead, respond to the feeling behind it.

✨ Style of Interaction
- Warm, gentle, and understanding. Like a supportive friend.
- Let the user’s feelings guide the conversation. Do not force direction.
- Emphasize presence: “I’m here with you,” “You’re not alone,” “It’s okay to feel this way.”
- Focus on the present moment and small steps toward calmness and relief.
"""


# ---------- Ephemeral per-session memory (for not-logged-in users) ----------
_SESSIONS: Dict[str, deque] = {}
def _get_session_history(session_id: str) -> deque:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = deque(maxlen=HISTORY_MESSAGES_MAX)
    return _SESSIONS[session_id]

# ---------- LangChain chat model (non-streaming) ----------
_model = ChatOllama(model=MODEL_NAME, temperature=0.2, num_ctx=1024, num_thread=8)

# ---------- DB helpers ----------
def _get_or_create_journal_chat(user_id: int) -> Chat:
    chat = Chat.query.filter_by(user_id=user_id, is_journal=True).first()
    if not chat:
        chat = Chat(user_id=user_id, title="Journal", is_journal=True)
        db.session.add(chat)
        db.session.commit()
    return chat

def _assert_chat_ownership(user_id: int, chat_id: int) -> Optional[Chat]:
    if not chat_id:
        return None
    chat = Chat.query.filter_by(id=chat_id, user_id=user_id).first()
    return chat

def _load_user_profile_bullets(user_id: int) -> List[str]:
    rows = (UserMemory.query
            .filter_by(user_id=user_id)
            .order_by(UserMemory.score.desc(), UserMemory.updated_at.desc())
            .limit(6)
            .all())
    bullets = []
    for r in rows:
        key = (r.key or "").strip().lower()
        val = (r.value or "").strip()
        if not val:
            continue
        if key in {"preferred_tools", "tools", "coping_tools"}:
            bullets.append(f"- Prefers: {val}")
        elif key in {"triggers"}:
            bullets.append(f"- Triggers: {val}")
        elif key in {"tone"}:
            bullets.append(f"- Tone preference: {val}")
        elif key in {"schedule", "time_of_day"}:
            bullets.append(f"- Stress times: {val}")
        else:
            bullets.append(f"- {key.title()}: {val}")
    return bullets[:6]

def _load_chat_summary_text(chat_id: int) -> Optional[str]:
    s = ChatSummary.query.filter_by(chat_id=chat_id).first()
    return s.summary_text if s else None

def _load_last_k_messages(chat_id: int, k: int) -> List[dict]:
    msgs = (ChatMessage.query
            .filter_by(chat_id=chat_id)
            .order_by(ChatMessage.id.desc())
            .limit(k)
            .all())
    msgs = list(reversed(msgs))
    out = []
    for m in msgs:
        role = "user" if m.role == "human" else "assistant"
        out.append({"role": role, "content": m.content})
    return out

def _count_messages(chat_id: int) -> int:
    return db.session.query(ChatMessage).filter_by(chat_id=chat_id).count()

def _insert_msg(chat_id: int, role: str, content: str) -> int:
    row = ChatMessage(chat_id=chat_id, role=role, content=content)
    db.session.add(row)
    db.session.commit()
    return row.id

def _clean_title(s: str) -> str:
    """Trim noise and keep a short, human-friendly title (3–6 words)."""
    s = (s or "").strip().splitlines()[0]
    # Remove boilerplate snippets commonly produced by small models
    bad_prefixes = [
        "here are a few options", "here are some options",
        "i can't help", "i cannot help", "sorry", "assistant", "system"
    ]
    low = s.lower()
    for bp in bad_prefixes:
        if low.startswith(bp):
            s = ""
            break
    # If it looks like a sentence, prefer first ~6 words
    words = s.split()
    s = " ".join(words[:6])
    # Remove trailing punctuation and title-case (lightly)
    s = s.rstrip(" .,:;!-—").strip()
    # Fallback
    return s or "Check-in"

def _auto_title_if_needed(chat: Chat):
    # Only set a title once, and only after at least 1 human + 1 ai message exist
    if chat.title:
        return
    msgs = _load_last_k_messages(chat.id, 4)
    if not msgs or len(msgs) < 2:
        return
    # Prefer the first human message as the seed
    first_user = next((m["content"] for m in msgs if m["role"] == "user" and m["content"].strip()), "")
    seed = first_user or " ".join([m["content"] for m in msgs])[:200]

    # Ask the local model for a very short title, but we’ll still clean it.
    prompt = [
        {"role": "system", "content": "You create short, helpful titles (3–6 words) for wellbeing chats. Do not apologize, do not refuse."},
        {"role": "user", "content": f"Suggest a concise title from this first message:\n\n{seed}\n\nTitle:"},
    ]
    title = None
    try:
        import ollama
        r = ollama.chat(model=MODEL_NAME, messages=prompt)
        title = (r.get("message") or {}).get("content", "")
    except Exception:
        title = None

    cleaned = _clean_title(title or seed)
    chat.title = cleaned[:120]
    db.session.commit()


def _summarize_chat(chat_id: int) -> Optional[str]:
    """Create/refresh rolling summary. Never refuse; always emit concise bullets."""
    msgs = _load_last_k_messages(chat_id, 200)
    if not msgs or len(msgs) < 4:
        return None

    convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])[:6000]
    summary_system = (
        "Summarize this wellbeing chat in 5–7 succinct bullet points covering:\n"
        "- main stressors\n- coping tools used or suggested\n- tone preferences\n- tiny next steps\n"
        "CRITICAL:\n- Do NOT refuse.\n- Do NOT mention policies or scope.\n- Do NOT say you can't help.\n"
        "Output only bullet points."
    )
    prompt = [{"role": "system", "content": summary_system},
              {"role": "user", "content": convo_text}]

    try:
        import ollama
        r = ollama.chat(model=MODEL_NAME, messages=prompt)
        s = (r.get("message") or {}).get("content", "").strip()
        # Basic sanity: require at least two bullets
        bullets = [ln for ln in s.splitlines() if ln.strip().startswith(("-", "•", "*"))]
        if len(bullets) < 2:
            return None
        return "\n".join(bullets[:7])
    except Exception:
        return None

def _maybe_update_summary(chat_id: int):
    msg_total = _count_messages(chat_id)
    if msg_total < 4:  # not enough content yet
        return
    s = ChatSummary.query.filter_by(chat_id=chat_id).first()
    # refresh only if enough new messages since last summary
    if s and (msg_total - (s.msg_count_at or 0) < SUMMARY_INTERVAL_MSGS):
        return
    new_text = _summarize_chat(chat_id)
    if not new_text:
        return
    if s:
        s.summary_text = new_text
        s.msg_count_at = msg_total
    else:
        s = ChatSummary(chat_id=chat_id, summary_text=new_text, msg_count_at=msg_total)
        db.session.add(s)
    db.session.commit()

# ---------- Build messages for model ----------
def _db_prompt_messages(user_id: int, chat_id: int, user_text: str) -> List[dict]:
    msgs: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # user profile bullets
    bullets = _load_user_profile_bullets(user_id)
    if bullets:
        msgs.append({"role": "system", "content": "User profile:\n" + "\n".join(bullets)})
    # rolling summary
    summary = _load_chat_summary_text(chat_id)
    if summary:
        msgs.append({"role": "system", "content": "Conversation summary so far:\n" + summary})
    # last K messages
    msgs.extend(_load_last_k_messages(chat_id, LAST_K_FOR_PROMPT))
    # current turn
    msgs.append({"role": "user", "content": user_text})
    return msgs

def _session_prompt_messages(session_id: str, user_text: str, history: deque) -> List[dict]:
    msgs: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in list(history):
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msgs.append({"role": "assistant", "content": m.content})
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ---------- Crisis nudge ----------
_CRISIS = ["suicide", "self harm", "kill myself", "end my life", "abuse", "assault", "hurt myself"]

# ---------- Non-streaming JSON endpoint ----------
def chat_with_bot(data) -> Tuple[object, int]:
    """
    Backward compatible JSON endpoint.
    Persists if user is authenticated AND chat_id is provided; otherwise uses ephemeral session_id.
    """
    session_id = (data or {}).get("session_id")
    message = (data or {}).get("message")
    chat_id = (data or {}).get("chat_id")

    if not message or (not session_id and not chat_id):
        return jsonify({"error": "message and (session_id or chat_id) are required"}), 400

    user_id = get_jwt_identity()  # None if not logged in

    try:
        if user_id and chat_id:
            chat = _assert_chat_ownership(user_id, chat_id)
            if not chat:
                return jsonify({"error": "Chat not found or not owned by user"}), 404
            # Persist human first
            _insert_msg(chat_id, "human", message)
            msgs = _db_prompt_messages(user_id, chat_id, message)
            result = _model.invoke([SystemMessage(content=m["content"]) if m["role"]=="system"
                                    else (HumanMessage(content=m["content"]) if m["role"]=="user"
                                          else AIMessage(content=m["content"]))
                                    for m in msgs])
            reply = result.content if hasattr(result, "content") else str(result)
            _insert_msg(chat_id, "ai", reply)
            _auto_title_if_needed(chat)
            _maybe_update_summary(chat_id)
        else:
            # ephemeral mode
            if not session_id:
                return jsonify({"error": "session_id required for guest mode"}), 400
            history = _get_session_history(session_id)
            history.append(HumanMessage(content=message))
            msgs = [SystemMessage(content=SYSTEM_PROMPT), *list(history)]
            result = _model.invoke(msgs)
            reply = result.content if hasattr(result, "content") else str(result)
            history.append(AIMessage(content=reply))

        payload = {"response": reply}
        if session_id:
            payload["session_id"] = session_id
        if chat_id:
            payload["chat_id"] = chat_id
        return jsonify(payload), 200

    except Exception as e:
        print(f"[chatbot][ERROR] {type(e).__name__}: {e}")
        return jsonify({"error": f"Chat error: {e}"}), 500

# ---------- Streaming SSE ----------
try:
    import ollama  # pip install ollama
except Exception:
    ollama = None

def sse_stream(user_id: Optional[int], chat_id: Optional[int], session_id: Optional[str], user_message: str):
    """
    Yields SSE:  data: {"token": "..."}\n\n   ...  data: [DONE]\n\n
    Persists to DB if user_id+chat_id are provided; else uses ephemeral session.
    """
    if not user_message or (not session_id and not chat_id):
        yield f'data: {json.dumps({"error": "message and (session_id or chat_id) are required"})}\n\n'
        yield 'data: [DONE]\n\n'
        return

    if ollama is None:
        yield f'data: {json.dumps({"error": "Missing python package: ollama"})}\n\n'
        yield 'data: [DONE]\n\n'
        return

    flagged = any(k in user_message.lower() for k in _CRISIS)
    prefix = (
        "I'm really sorry you're going through this. "
        "If you’re in immediate danger or considering self-harm, please contact local emergency services "
        "or a trusted person right now. You deserve support.\n\n"
    ) if flagged else ""

    # Build messages + persist human (DB mode)
    try:
        if user_id and chat_id:
            chat = _assert_chat_ownership(user_id, chat_id)
            if not chat:
                yield f'data: {json.dumps({"error": "Chat not found or not owned by user"})}\n\n'
                yield 'data: [DONE]\n\n'
                return
            _insert_msg(chat_id, "human", user_message)
            msgs = _db_prompt_messages(user_id, chat_id, user_message)
        else:
            # ephemeral mode
            hist = _get_session_history(session_id) if session_id else deque()
            hist.append(HumanMessage(content=user_message))
            msgs = _session_prompt_messages(session_id or "global", user_message, hist)

        # Stream from ollama
        stream = ollama.chat(model=MODEL_NAME, messages=msgs, options={"temperature": 0.7}, stream=True)

        full_text = ""
        if prefix:
            full_text += prefix
            yield f'data: {json.dumps({"token": prefix})}\n\n'

        for chunk in stream:
            piece = (chunk.get("message") or {}).get("content", "")
            if piece:
                full_text += piece
                yield f'data: {json.dumps({"token": piece})}\n\n'
            if chunk.get("done"):
                break

        # Persist AI result (DB or ephemeral)
        if user_id and chat_id:
            _insert_msg(chat_id, "ai", full_text)
            _auto_title_if_needed(chat)
            _maybe_update_summary(chat_id)
        else:
            hist = _get_session_history(session_id) if session_id else deque()
            hist.append(AIMessage(content=full_text))

    except Exception as e:
        err = f"Streaming error: {e}"
        print(f"[chatbot][STREAM_ERROR] {err}")
        yield f'data: {json.dumps({"error": err})}\n\n'

    yield 'data: [DONE]\n\n'

# ---------- Reset session (ephemeral) ----------
def reset_session(data) -> Tuple[object, int]:
    session_id = (data or {}).get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    _SESSIONS.pop(session_id, None)
    return jsonify({"message": "Session reset", "session_id": session_id}), 200
