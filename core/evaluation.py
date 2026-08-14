"""The evaluation contract: what an LLM is asked for, and what is accepted back.

Deliberately absent from :class:`TaskEvaluation`: the workflow decision. The
model supplies *analysis* - how well the task matches, how long it will take,
what could go wrong - and the decision engine turns that into SKIP /
REVIEW_REQUIRED / HIGH_MATCH using rules that live in code and can be reasoned
about without re-running a model.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.enums import Complexity
from core.exceptions import EvaluationError
from core.profile import SkillProfile
from core.schemas import Task


class TaskEvaluation(BaseModel):
    """Structured analysis of a single task. Never trusted unvalidated."""

    model_config = ConfigDict(extra="ignore")  # tolerate extra keys from chatty models

    score: int = Field(ge=0, le=100)
    skill_match: int = Field(ge=0, le=100)
    deadline_feasibility: int = Field(default=100, ge=0, le=100)
    estimated_hours: float = Field(default=1.0, gt=0, le=1000)
    complexity: Complexity = Complexity.MEDIUM
    missing_skills: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    reason: str = ""

    #: Which provider produced this, for the audit trail.
    provider: str = "unknown"
    model: str | None = None

    @field_validator("missing_skills", "risks", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> Any:
        # Models sometimes return a single string where a list was asked for.
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator("complexity", mode="before")
    @classmethod
    def _coerce_complexity(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value


def extract_json_object(raw: str) -> dict[str, Any]:
    """Pull the first complete JSON object out of a model response.

    Models wrap JSON in prose or code fences no matter how firmly they are
    told not to, so this scans for balanced braces rather than assuming the
    whole response parses.
    """
    if not raw or not raw.strip():
        raise EvaluationError("empty response from provider")

    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag) and the closer.
        text = text.split("\n", 1)[-1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        raise EvaluationError(f"no JSON object in response: {raw[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"malformed JSON in response: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise EvaluationError("response JSON was not an object")
                return parsed

    raise EvaluationError("unterminated JSON object in response")


def parse_evaluation(raw: str) -> TaskEvaluation:
    """Extract and validate a :class:`TaskEvaluation` from raw model text."""
    payload = extract_json_object(raw)
    try:
        return TaskEvaluation.model_validate(payload)
    except Exception as exc:
        raise EvaluationError(f"response did not match the evaluation schema: {exc}") from exc


SYSTEM_PROMPT = """You evaluate whether a freelance/contract technical task \
is a good fit for one specific contributor. You return analysis only. You do \
not decide whether to accept the task; a separate rules engine does that.

Respond with a single JSON object and nothing else. No prose, no code fences.

Schema:
{
  "score": integer 0-100, overall fit,
  "skill_match": integer 0-100, how well their skills cover the requirements,
  "deadline_feasibility": integer 0-100, 100 if no deadline pressure,
  "estimated_hours": number, realistic hours for this contributor,
  "complexity": "LOW" | "MEDIUM" | "HIGH",
  "missing_skills": array of strings, empty if none,
  "risks": array of strings, e.g. unclear requirements or scope creep,
  "reason": string, two sentences at most
}

Be calibrated, not generous. A task needing skills they lack scores below 50 \
however interesting it is. Unclear requirements are a risk, not a reason to \
guess high."""


def build_user_prompt(task: Task, profile: SkillProfile, *, capacity_note: str = "") -> str:
    """Render one task plus the contributor profile into a prompt."""
    requirements = "\n".join(f"  - {r}" for r in task.requirements) or "  (none listed)"
    strengths = ", ".join(profile.all_strengths) or "(none recorded)"
    avoid = ", ".join(profile.avoid) or "(nothing recorded)"
    deadline = task.deadline.isoformat() if task.deadline else "none stated"

    sections = [
        "CONTRIBUTOR",
        f"  Name: {profile.display_name}",
        f"  Headline: {profile.headline or '(none)'}",
        f"  Strengths: {strengths}",
        f"  Prefers to avoid: {avoid}",
        f"  Available hours per week: {profile.weekly_hours}",
        f"  Notes: {profile.notes or '(none)'}",
        "",
        "TASK",
        f"  Project: {task.project_name}",
        f"  Title: {task.title}",
        f"  Deadline: {deadline}",
        f"  Reward: {task.reward or 'not stated'}",
        "  Requirements:",
        requirements,
        "  Description:",
        f"  {task.description or '(none provided)'}",
    ]
    if capacity_note:
        sections += ["", "CURRENT WORKLOAD", f"  {capacity_note}"]
    sections += ["", "Return the JSON object now."]
    return "\n".join(sections)


REPAIR_SUFFIX = (
    "\n\nYour previous response could not be parsed. Return ONLY a single valid "
    "JSON object matching the schema exactly. No explanation, no code fences."
)
