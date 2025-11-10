# backend/services/notifier_service.py
import os, smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime
from itsdangerous import URLSafeSerializer
import requests

APP_SECRET = os.environ["APP_SECRET"]
DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

def sign_action_token(subtask_id: str, action: str, expires_ts: datetime):
    s = URLSafeSerializer(APP_SECRET)
    # encode ISO string for exp
    token = s.dumps({"sid": subtask_id, "a": action, "exp": expires_ts.isoformat()})
    return token

def verify_action_token(token: str):
    from itsdangerous import BadSignature
    s = URLSafeSerializer(APP_SECRET)
    try:
        data = s.loads(token)
        # simple exp check
        from datetime import datetime
        if "exp" in data and datetime.fromisoformat(data["exp"]) < datetime.utcnow():
            return None
        return data
    except BadSignature:
        return None

def _magic_link(base_url: str, token: str, action: str, extra: str = ""):
    return f"{base_url}/api/focus/hook?token={token}&action={action}{extra}"

def send_email(to_email: str, subject: str, html_body: str):
    msg = MIMEText(html_body, "html")
    display_from = EMAIL_FROM or SMTP_USER
    msg["From"] = formataddr(("Focus Companion", display_from))
    msg["To"] = to_email
    msg["Subject"] = subject
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(display_from, [to_email], msg.as_string())

def send_telegram(chat_id: str, text: str, buttons=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode":"HTML"}
    if buttons:
        payload["reply_markup"] = {"inline_keyboard": buttons}
    requests.post(url, json=payload)

def render_email_for_subtask(subtask, base_url: str, to_email: str):
    # create action tokens with short expiry e.g. +8h
    from datetime import timedelta
    exp = datetime.utcnow() + timedelta(hours=8)
    t_start = sign_action_token(subtask["id"], "start", exp)
    t_done  = sign_action_token(subtask["id"], "done", exp)
    t_snooz = sign_action_token(subtask["id"], "snooze", exp)
    t_block = sign_action_token(subtask["id"], "blocked", exp)

    start_link = _magic_link(base_url, t_start, "start")
    done_link  = _magic_link(base_url, t_done, "done")
    snooze15   = _magic_link(base_url, t_snooz, "snooze", "&min=15")
    blocked    = _magic_link(base_url, t_block, "blocked")

    subject = f"⏰ Time for: {subtask['title']}"
    html = f"""
    <div style="font-family:Inter,Arial,sans-serif">
      <h3>⏰ Start: {subtask['title']}</h3>
      <p><b>Definition of Done:</b> {subtask['dod_text']}</p>
      <p><b>Estimate:</b> {subtask['estimate_min']} min</p>
      <p>
        <a href="{start_link}">Start</a> ·
        <a href="{done_link}">Done</a> ·
        <a href="{snooze15}">Snooze 15m</a> ·
        <a href="{blocked}">Blocked</a>
      </p>
      <hr/>
      <small>These links work without login and expire in 8 hours.</small>
    </div>
    """
    return subject, html
