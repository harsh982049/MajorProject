import os
import json
import re
from typing import List, Dict, Any, Annotated
from pydantic import BaseModel, Field, ValidationError

import google.generativeai as genai

# ---------- Config ----------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # safe default

# ---------- Schemas ----------
class Step(BaseModel):
    idx: Annotated[int, Field(ge=1)]
    title: Annotated[str, Field(min_length=3)]
    definition_of_done: Annotated[str, Field(min_length=3)]
    estimate_min: Annotated[int, Field(gt=0, le=60)]

class Plan(BaseModel):
    steps: list[Step]

class MicroStep(BaseModel):
    title: Annotated[str, Field(min_length=3)]
    definition_of_done: Annotated[str, Field(min_length=3)]
    estimate_min: Annotated[int, Field(gt=0, le=10)]

class MicroPlan(BaseModel):
    micro_steps: list[MicroStep]


# ---------- System prompts ----------
PLANNER_SYSTEM = """You are a precise planning assistant.
Return STRICT JSON ONLY with the shape:
{
  "steps": [
    {"idx":1, "title":"...", "definition_of_done":"...", "estimate_min":N},
    ...
  ]
}
Rules:
- 3–5 atomic, verifiable steps with clear definitions of done.
- Concrete verbs (collect, draft, review, submit).
- Keep total estimate <= timebox_min (+10% slack).
- idx starts at 1 and increments by 1.
"""

MICRO_SYSTEM = """You split blocked tasks into 2–3 micro-steps (<=10 min each).
Return STRICT JSON ONLY:
{
  "micro_steps": [
    {"title":"...", "definition_of_done":"...", "estimate_min":N},
    ...
  ]
}
"""

# ---------- Helpers ----------
def _extract_json(text: str) -> Dict[str, Any]:
    """Try plain load, else extract the first {...} block."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        return {}

def _make_model(system_instruction: str):
    """Create a GenerativeModel with system_instruction (no 'system' role in messages)."""
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
        generation_config={"temperature": 0.3}
    )

# ---------- Planner ----------
def plan_subtasks(goal: str, timebox_min: int, constraints: Dict[str, Any]) -> Plan:
    user_prompt = (
        f"Goal: {goal}\n"
        f"timebox_min: {timebox_min}\n"
        f"constraints: {constraints}"
    )
    model = _make_model(PLANNER_SYSTEM)

    # 1st attempt
    resp = model.generate_content(user_prompt)
    data = _extract_json(getattr(resp, "text", ""))

    # validate or repair once
    try:
        plan = Plan.model_validate(data)
    except ValidationError as e:
        err = e.errors()
        repair_model = _make_model(PLANNER_SYSTEM)
        repair_prompt = user_prompt + f"\nThe previous JSON was invalid: {err}. Return corrected STRICT JSON only."
        resp2 = repair_model.generate_content(repair_prompt)
        data2 = _extract_json(getattr(resp2, "text", ""))
        plan = Plan.model_validate(data2)

    # enforce limits
    steps = plan.steps[:5]
    return Plan(steps=steps)

# ---------- Micro-splitter ----------
def micro_split(title: str, dod: str, notes: str = "", remaining_time_min: int = 30) -> MicroPlan:
    user_prompt = (
        "Blocked subtask:\n"
        f"title: {title}\n"
        f"definition_of_done: {dod}\n"
        f"notes_from_user: {notes}\n"
        f"remaining_time_min: {remaining_time_min}"
    )
    model = _make_model(MICRO_SYSTEM)
    resp = model.generate_content(user_prompt)
    data = _extract_json(getattr(resp, "text", ""))

    # validate or repair once
    try:
        mp = MicroPlan.model_validate(data)
    except ValidationError as e:
        err = e.errors()
        repair_model = _make_model(MICRO_SYSTEM)
        repair_prompt = user_prompt + f"\nThe previous JSON was invalid: {err}. Return corrected STRICT JSON only."
        resp2 = repair_model.generate_content(repair_prompt)
        data2 = _extract_json(getattr(resp2, "text", ""))
        mp = MicroPlan.model_validate(data2)

    # clamp to 3 micro-steps max
    mp.micro_steps = mp.micro_steps[:3]
    return mp
