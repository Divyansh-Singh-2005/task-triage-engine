"""A deterministic, offline provider.

Exists for three reasons: the pipeline must be runnable and testable with no
API key, the test suite must not depend on a network call, and a rule-based
baseline makes it obvious when a model's scores drift somewhere strange.

The scoring is intentionally crude. It is a floor, not a replacement.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from core.enums import Complexity
from core.evaluation import TaskEvaluation
from core.profile import SkillProfile
from core.schemas import Task
from integrations.llm.base import LLMProvider

_WORD = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


class HeuristicProvider(LLMProvider):
    """Scores by keyword overlap between the task and the profile."""

    name = "heuristic"
    model = None

    def evaluate_task(
        self, task: Task, profile: SkillProfile, *, capacity_note: str = ""
    ) -> TaskEvaluation:
        haystack = _tokens(
            " ".join([task.title, task.description, " ".join(task.requirements)])
        )

        strengths = profile.all_strengths
        matched = [s for s in strengths if _tokens(s) & haystack]
        skill_match = round(100 * len(matched) / len(strengths)) if strengths else 50

        # Requirements the profile does not visibly cover.
        missing = [r for r in task.requirements if not (_tokens(r) & _tokens(" ".join(strengths)))]
        if task.requirements:
            coverage = 100 - round(100 * len(missing) / len(task.requirements))
            skill_match = round(0.5 * skill_match + 0.5 * coverage)

        complexity, hours = self._estimate(task)
        feasibility = self._feasibility(task, hours, profile)

        score = round(0.55 * skill_match + 0.30 * feasibility + 15)
        if any(_tokens(term) & haystack for term in profile.avoid):
            score = max(0, score - 30)
        score = max(0, min(100, score))

        risks: list[str] = []
        if not task.requirements:
            risks.append("No requirements listed; scope is unclear")
        if len(task.description) < 80:
            risks.append("Description is thin; effort estimate is low confidence")
        if missing:
            risks.append(f"{len(missing)} requirement(s) not covered by the profile")

        return TaskEvaluation(
            score=score,
            skill_match=skill_match,
            deadline_feasibility=feasibility,
            estimated_hours=hours,
            complexity=complexity,
            missing_skills=missing,
            risks=risks,
            reason=(
                f"Keyword baseline: matched {len(matched)}/{len(strengths) or 0} profile "
                f"strengths, {len(missing)} requirement(s) uncovered."
            ),
            provider=self.name,
        )

    @staticmethod
    def _estimate(task: Task) -> tuple[Complexity, float]:
        weight = len(task.requirements) + len(task.description) / 400
        if weight <= 2:
            return Complexity.LOW, 3.0
        if weight <= 5:
            return Complexity.MEDIUM, 8.0
        return Complexity.HIGH, 20.0

    @staticmethod
    def _feasibility(task: Task, hours: float, profile: SkillProfile) -> int:
        if task.deadline is None:
            return 100
        days_left = (task.deadline - datetime.now(timezone.utc)).total_seconds() / 86400
        if days_left <= 0:
            return 0
        available = profile.weekly_hours * (days_left / 7)
        if available <= 0:
            return 0
        return max(0, min(100, round(100 * available / hours)))
