"""The LLM provider boundary.

Two levels, because not every provider is a chat model:

* :class:`LLMProvider` - anything that can turn a task into a
  :class:`TaskEvaluation`. This is the only type the rest of the app depends on.
* :class:`TextLLMProvider` - the common case: build a prompt, send text, parse
  JSON back. Concrete chat providers subclass this and implement ``complete``.

A rule-based provider that never calls a model subclasses the first directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.evaluation import (
    REPAIR_SUFFIX,
    SYSTEM_PROMPT,
    TaskEvaluation,
    build_user_prompt,
    parse_evaluation,
)
from core.exceptions import EvaluationError
from core.profile import SkillProfile
from core.schemas import Task


class LLMProvider(ABC):
    """Turns a task into structured analysis, by whatever means."""

    name: str = "abstract"
    model: str | None = None

    @abstractmethod
    def evaluate_task(
        self, task: Task, profile: SkillProfile, *, capacity_note: str = ""
    ) -> TaskEvaluation:
        """Analyze one task. Raise :class:`EvaluationError` on failure."""

    def describe(self) -> str:
        return f"{self.name}({self.model})" if self.model else self.name


class TextLLMProvider(LLMProvider):
    """Prompt-and-parse provider for chat models.

    One retry with an explicit repair instruction, because unparseable output
    is usually a formatting slip that a second attempt fixes, and paying for
    two calls beats dropping the task on the floor.
    """

    max_attempts: int = 2

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        """Send a prompt, return raw text."""

    def evaluate_task(
        self, task: Task, profile: SkillProfile, *, capacity_note: str = ""
    ) -> TaskEvaluation:
        prompt = build_user_prompt(task, profile, capacity_note=capacity_note)
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            attempt_prompt = prompt if attempt == 1 else prompt + REPAIR_SUFFIX
            try:
                raw = self.complete(SYSTEM_PROMPT, attempt_prompt)
                evaluation = parse_evaluation(raw)
            except EvaluationError as exc:
                last_error = exc
                continue
            return evaluation.model_copy(
                update={"provider": self.name, "model": self.model}
            )

        raise EvaluationError(
            f"{self.name} failed to return a valid evaluation after "
            f"{self.max_attempts} attempts: {last_error}"
        )
