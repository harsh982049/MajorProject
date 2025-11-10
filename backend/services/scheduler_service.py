from datetime import datetime, timedelta, timezone
import pytz

def _parse_hhmm(hhmm: str):
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

def schedule_subtasks(now_utc: datetime, tz_name: str, work_start_hhmm: str, work_end_hhmm: str,
                      buffer_min: int, subtasks):
    """
    Returns subtasks with aware UTC datetimes for planned_start_ts / planned_end_ts,
    never in the past relative to now_utc.
    """
    # Ensure 'now_utc' is aware UTC
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    tz = pytz.timezone(tz_name or "Asia/Kolkata")
    now_local = now_utc.astimezone(tz)

    wh, wm = _parse_hhmm(work_start_hhmm)
    eh, em = _parse_hhmm(work_end_hhmm)

    def local_day_start(d):
        return d.replace(hour=wh, minute=wm, second=0, microsecond=0)
    def local_day_end(d):
        return d.replace(hour=eh, minute=em, second=0, microsecond=0)

    # Start from max(now, work start), rounded up to next minute
    start_local = max(now_local, local_day_start(now_local))
    if start_local.second or start_local.microsecond:
        start_local = (start_local + timedelta(minutes=1)).replace(second=0, microsecond=0)

    if start_local > local_day_end(now_local):
        start_local = local_day_start(now_local + timedelta(days=1))

    scheduled = []
    cursor = start_local
    for st in subtasks:
        est = timedelta(minutes=int(st["estimate_min"]))

        # Switch to next day if no room today
        if cursor + est > local_day_end(cursor):
            cursor = local_day_start(cursor + timedelta(days=1))

        # Final “no past” clamp for races
        if cursor < now_local:
            cursor = (now_local + timedelta(minutes=1)).replace(second=0, microsecond=0)

        start_dt_local = cursor
        end_dt_local = cursor + est
        cursor = end_dt_local + timedelta(minutes=buffer_min)

        st["planned_start_ts"] = start_dt_local.astimezone(timezone.utc)
        st["planned_end_ts"]   = end_dt_local.astimezone(timezone.utc)
        scheduled.append(st)

    return scheduled
