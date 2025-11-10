# backend/services/hooks_service.py
from datetime import datetime, timedelta
from typing import Any, Dict
from services.supabase_client import supabase
from services.focus_planner_service import micro_split
from services.scheduler_service import schedule_subtasks
from services.notifier_service import send_email, render_email_for_subtask, verify_action_token
import os, pytz

BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:5000")

def _get_subtask(subtask_id: str):
    return supabase.table("subtasks").select("*").eq("id", subtask_id).single().execute().data

def _get_task(task_id: str):
    return supabase.table("tasks").select("*").eq("id", task_id).single().execute().data

def _get_user_and_prefs(user_id: str):
    user = supabase.table("users").select("*").eq("id", user_id).single().execute().data
    prefs = supabase.table("user_prefs").select("*").eq("user_id", user_id).single().execute().data
    return user, prefs

def _to_iso(v):
    return v.isoformat() if isinstance(v, datetime) else v

def _coerce_dt(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_iso(v) for k, v in d.items()}

def _update_subtask_state(subtask_id: str, new_state: str, extra_fields: Dict[str, Any] | None = None):
    fields = {"state": new_state}
    if extra_fields:
        fields.update(_coerce_dt(extra_fields))  # <-- ensure ISO strings
    return supabase.table("subtasks").update(fields).eq("id", subtask_id).execute()

def _insert_audit(subtask_id, task_id, event_type, payload):
    supabase.table("audit_events").insert({
        "subtask_id": subtask_id, "task_id": task_id,
        "event_type": event_type, "payload": payload
    }).execute()

def handle_magic_hook(token: str, action: str, query_params: dict):
    data = verify_action_token(token)
    if not data:
        return False, "Invalid or expired link."
    subtask_id = data["sid"]
    st = _get_subtask(subtask_id)
    if not st:
        return False, "Subtask not found."

    if action == "start":
        now = datetime.utcnow()
        _update_subtask_state(subtask_id, "in_progress", {"actual_start_ts": now})
        _insert_audit(subtask_id, st["task_id"], "user_action", {"action":"start"})
        return True, "Marked as In Progress."

    if action == "done":
        now = datetime.utcnow()
        _update_subtask_state(subtask_id, "done", {"actual_end_ts": now})
        _insert_audit(subtask_id, st["task_id"], "user_action", {"action":"done"})
        # (Optional) pull next scheduled subtask earlier (simple policy omitted in MVP)
        return True, "Great! Marked Done."

    if action == "snooze":
        mins = int(query_params.get("min", 15))
        # Move planned start/end by mins and shift others in same task forward (simple: only this one)
        ps = datetime.fromisoformat(st["planned_start_ts"].replace("Z","")) if st["planned_start_ts"] else datetime.utcnow()
        pe = datetime.fromisoformat(st["planned_end_ts"].replace("Z",""))   if st["planned_end_ts"] else ps + timedelta(minutes=st["estimate_min"])
        ps2, pe2 = ps + timedelta(minutes=mins), pe + timedelta(minutes=mins)
        _update_subtask_state(subtask_id, "snoozed", {
            "planned_start_ts": ps2.isoformat(), "planned_end_ts": pe2.isoformat()
        })
        _insert_audit(subtask_id, st["task_id"], "user_action", {"action":"snooze","minutes":mins})
        return True, f"Snoozed by {mins} min."

    if action == "blocked":
        # Micro-split and reschedule
        task = _get_task(st["task_id"])
        user, prefs = _get_user_and_prefs(task["user_id"])
        micro = micro_split(title=st["title"], dod=st["dod_text"])
        # Close current as blocked
        _update_subtask_state(subtask_id, "blocked", {"actual_end_ts": datetime.utcnow().isoformat()})
        _insert_audit(subtask_id, st["task_id"], "user_action", {"action":"blocked"})

        # Insert micro-steps with sequential indices after current idx
        next_idx = st["idx"]
        to_insert = []
        for i, m in enumerate(micro.micro_steps, start=1):
            to_insert.append({
                "task_id": st["task_id"],
                "idx": next_idx + i - 1,
                "title": m.title,
                "dod_text": m.definition_of_done,
                "estimate_min": m.estimate_min,
                "state": "scheduled"
            })
        supabase.table("subtasks").insert(to_insert).execute()

        # Pull all remaining (scheduled/snoozed) subtasks for this task, order by idx, and reschedule greedily
        rem = supabase.table("subtasks").select("*").eq("task_id", st["task_id"]).in_("state", ["scheduled","snoozed"]).order("idx").execute().data

        now_utc = datetime.utcnow()
        scheduled = schedule_subtasks(
            now_utc=now_utc,
            tz_name=prefs["timezone"],
            work_start_hhmm=prefs["work_start_hhmm"],
            work_end_hhmm=prefs["work_end_hhmm"],
            buffer_min=prefs["default_buffer_min"],
            subtasks=rem
        )

        # Persist planned times + create notifications for the first new micro-step
        for r in scheduled:
            supabase.table("subtasks").update({
                "planned_start_ts": r["planned_start_ts"].isoformat(),
                "planned_end_ts": r["planned_end_ts"].isoformat()
            }).eq("id", r["id"]).execute()

        # Send confirmation for the very next scheduled subtask via email (if enabled)
        if user.get("email"):
            # find next subtask by earliest planned_start
            nxt = sorted(scheduled, key=lambda x: x["planned_start_ts"])[0] if scheduled else None
            if nxt:
                from services.notifier_service import render_email_for_subtask, send_email, sign_action_token
                subject, html = render_email_for_subtask(nxt, BASE_URL, user["email"])
                send_email(user["email"], subject, html)

        return True, "Blocked acknowledged; micro-steps created and rescheduled."

    return False, "Unknown action."
