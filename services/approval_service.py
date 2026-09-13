"""Human approval: the transitions a person can trigger by hand.

Approval is where a machine decision becomes a commitment, so every
transition here is validated against the task's current status rather than
trusting whatever the UI thought it was showing. A dashboard tab left open
for an hour will otherwise happily approve a task that has since been skipped.

Note what this module does *not* do: it never contacts any external platform.
Approving marks intent locally. Acting on that intent is a separate concern
and remains unbuilt pending confirmation of what automation is permitted.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from core.exceptions import AgentError
from core.enums import TaskStatus
from core.schemas import TaskRules
from agent.workload_manager import WorkloadManager
from database.models import TaskRecord
from database.repositories import EventRepository, TaskRepository


class ApprovalError(AgentError):
    """A requested transition is not legal from the task's current status."""


#: Statuses a person may approve from. DISCOVERED is excluded on purpose:
#: approving something that was never evaluated skips the whole point.
APPROVABLE_FROM = (TaskStatus.REVIEW_REQUIRED,)
REJECTABLE_FROM = (TaskStatus.REVIEW_REQUIRED, TaskStatus.APPROVED, TaskStatus.DISCOVERED)
RESETTABLE_FROM = (TaskStatus.SKIPPED, TaskStatus.FAILED, TaskStatus.REVIEW_REQUIRED)


class ApprovalService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.tasks = TaskRepository(session)
        self.events = EventRepository(session)
        self.workload = WorkloadManager(self.tasks)

    def _get(self, task_id: int) -> TaskRecord:
        record = self.session.get(TaskRecord, task_id)
        if record is None:
            raise ApprovalError(f"task {task_id} does not exist")
        return record

    def approve(self, task_id: int, rules: TaskRules, *, force: bool = False) -> TaskRecord:
        """Mark a reviewed task as approved.

        Capacity is re-checked here even though the decision engine already
        checked it, because time passes between the two and the person may
        have approved something else in between. ``force`` is the deliberate
        override, and it is recorded as such.
        """
        record = self._get(task_id)
        if record.status not in APPROVABLE_FROM:
            raise ApprovalError(
                f"cannot approve {record.external_task_id} from {record.status.value}; "
                f"expected one of {', '.join(s.value for s in APPROVABLE_FROM)}"
            )

        status = self.workload.status(rules)
        if not status.has_capacity and not force:
            raise ApprovalError(
                f"workload is full ({status.note()}); raise the limit or approve with force"
            )

        self.tasks.set_status(record, TaskStatus.APPROVED)
        self.events.log(
            "task_approved",
            message=f"approved by user{' (forced over capacity)' if force and not status.has_capacity else ''}",
            task_id=record.id,
            project_name=record.project.name,
            payload={"forced": bool(force and not status.has_capacity)},
        )
        return record

    def reject(self, task_id: int, *, reason: str = "") -> TaskRecord:
        """Skip a task by hand."""
        record = self._get(task_id)
        if record.status not in REJECTABLE_FROM:
            raise ApprovalError(
                f"cannot skip {record.external_task_id} from {record.status.value}"
            )
        self.tasks.set_status(record, TaskStatus.SKIPPED)
        self.events.log(
            "task_rejected",
            message=reason or "skipped by user",
            task_id=record.id,
            project_name=record.project.name,
        )
        return record

    def reset(self, task_id: int) -> TaskRecord:
        """Send a task back to DISCOVERED so it gets evaluated again.

        Useful after editing the skill profile or the thresholds - the old
        evaluation rows stay, so the score history remains auditable.
        """
        record = self._get(task_id)
        if record.status not in RESETTABLE_FROM:
            raise ApprovalError(
                f"cannot reset {record.external_task_id} from {record.status.value}"
            )
        self.tasks.set_status(record, TaskStatus.DISCOVERED)
        self.events.log(
            "task_reset",
            message="queued for re-evaluation",
            task_id=record.id,
            project_name=record.project.name,
        )
        return record
