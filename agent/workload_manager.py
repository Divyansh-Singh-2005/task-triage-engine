"""Workload capacity.

Separate from the decision engine because "am I full?" is a question about
stored state, while "is this task good enough?" is a question about one
evaluation. Keeping them apart means the engine stays a pure function.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schemas import TaskRules
from database.repositories import TaskRepository


@dataclass(frozen=True)
class WorkloadStatus:
    active: int
    maximum: int

    @property
    def remaining(self) -> int:
        return max(0, self.maximum - self.active)

    @property
    def has_capacity(self) -> bool:
        return self.remaining > 0

    def note(self) -> str:
        """One line for the evaluation prompt and the audit trail."""
        state = "at capacity" if not self.has_capacity else f"{self.remaining} slot(s) free"
        return f"{self.active} of {self.maximum} active tasks; {state}"


class WorkloadManager:
    def __init__(self, tasks: TaskRepository) -> None:
        self.tasks = tasks

    def status(self, rules: TaskRules) -> WorkloadStatus:
        return WorkloadStatus(active=self.tasks.count_active(), maximum=rules.maximum_active_tasks)
