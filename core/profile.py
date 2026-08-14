"""The skill profile: who the agent is evaluating tasks *for*.

Kept in its own JSON file rather than in agent config, because it changes on a
completely different cadence - modes and thresholds get tuned weekly, a skill
profile maybe twice a year - and because it is the one input that makes
evaluation scores meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.exceptions import ConfigError


class SkillProfile(BaseModel):
    """What the person is good at, wants, and has time for."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = "Contributor"
    headline: str = ""
    skills: list[str] = Field(default_factory=list)
    preferred_technologies: list[str] = Field(default_factory=list)
    #: Topics to score down hard. Not a hard filter - a task can still surface
    #: for review if everything else about it is strong.
    avoid: list[str] = Field(default_factory=list)
    weekly_hours: float = Field(default=15.0, gt=0)
    notes: str = ""

    @property
    def all_strengths(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in list(self.skills) + list(self.preferred_technologies):
            key = item.strip().casefold()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out


def load_profile(path: str | Path) -> SkillProfile:
    """Read a profile, falling back to defaults when the file is absent."""
    p = Path(path)
    if not p.exists():
        return SkillProfile()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p} is not valid JSON: {exc}") from exc
    try:
        return SkillProfile.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"invalid skill profile in {p}: {exc}") from exc
