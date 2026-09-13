"""Repositories: the only place that writes SQL-ish queries.

Services depend on these, never on the session directly, so swapping the
storage backend later touches one layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.enums import Decision, TaskStatus
from core.naming import normalize_project_name, project_key
from core.schemas import Task
from database.models import AgentEvent, Evaluation, Project, TaskRecord


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_name(self, name: str) -> Project | None:
        key = project_key(name)
        return self.session.scalar(select(Project).where(Project.key == key))

    def get_or_create(self, name: str) -> Project:
        """Look up by canonical key, inserting only if genuinely new."""
        display = normalize_project_name(name)
        existing = self.get_by_name(display)
        if existing is not None:
            return existing
        project = Project(name=display, key=project_key(display))
        self.session.add(project)
        self.session.flush()  # assign the PK without ending the transaction
        return project

    def list_all(self) -> Sequence[Project]:
        return self.session.scalars(select(Project).order_by(Project.name)).all()

    def set_active(self, name: str) -> Project:
        """Mark one project active, clearing the flag from all others."""
        target = self.get_or_create(name)
        for project in self.list_all():
            project.is_active = project.id == target.id
        self.session.flush()
        return target


class TaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_external_id(self, source: str, external_task_id: str) -> TaskRecord | None:
        return self.session.scalar(
            select(TaskRecord).where(
                TaskRecord.source == source,
                TaskRecord.external_task_id == external_task_id,
            )
        )

    def get_by_fingerprint(self, fingerprint: str) -> TaskRecord | None:
        return self.session.scalar(
            select(TaskRecord).where(TaskRecord.fingerprint == fingerprint)
        )

    def find_duplicate(self, task: Task) -> TaskRecord | None:
        """Both duplicate guards in the order they should be trusted."""
        by_id = self.get_by_external_id(task.source, task.task_id)
        if by_id is not None:
            return by_id
        return self.get_by_fingerprint(task.fingerprint)

    def add(self, task: Task, project: Project) -> TaskRecord:
        record = TaskRecord(
            source=task.source,
            external_task_id=task.task_id,
            fingerprint=task.fingerprint,
            project_id=project.id,
            title=task.title,
            description=task.description,
            requirements=list(task.requirements),
            deadline=task.deadline,
            reward=task.reward,
            url=task.url,
            status=task.status,
            discovered_at=task.discovered_at,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def list_by_status(self, *statuses: TaskStatus) -> Sequence[TaskRecord]:
        stmt = select(TaskRecord).order_by(TaskRecord.discovered_at.desc())
        if statuses:
            stmt = stmt.where(TaskRecord.status.in_(statuses))
        return self.session.scalars(stmt).all()

    def count_active(self) -> int:
        """Tasks that occupy a workload slot.

        Approved and claim-pending count, because the commitment is already
        made even though no work has started.
        """
        active = (TaskStatus.APPROVED, TaskStatus.CLAIM_PENDING, TaskStatus.CLAIMED)
        return len(self.list_by_status(*active))

    def set_status(self, record: TaskRecord, status: TaskStatus) -> TaskRecord:
        record.status = status
        self.session.flush()
        return record


class EvaluationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(
        self,
        task: TaskRecord,
        *,
        score: int,
        decision: Decision,
        reason: str = "",
        risks: list[str] | None = None,
        estimated_hours: float | None = None,
        skill_match: int | None = None,
        deadline_feasibility: int | None = None,
        complexity=None,
        model: str | None = None,
    ) -> Evaluation:
        evaluation = Evaluation(
            task_id=task.id,
            score=score,
            decision=decision,
            reason=reason,
            risks=list(risks or []),
            estimated_hours=estimated_hours,
            skill_match=skill_match,
            deadline_feasibility=deadline_feasibility,
            complexity=complexity,
            model=model,
        )
        self.session.add(evaluation)
        self.session.flush()
        return evaluation

    def latest_for(self, task_id: int) -> Evaluation | None:
        return self.session.scalar(
            select(Evaluation)
            .where(Evaluation.task_id == task_id)
            .order_by(Evaluation.created_at.desc(), Evaluation.id.desc())
        )


class EventRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def log(
        self,
        event_type: str,
        *,
        message: str = "",
        task_id: int | None = None,
        project_name: str | None = None,
        payload: dict | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            event_type=event_type,
            message=message,
            task_id=task_id,
            project_name=project_name,
            payload=payload or {},
        )
        self.session.add(event)
        self.session.flush()
        return event

    def recent(self, limit: int = 50, since: datetime | None = None) -> Sequence[AgentEvent]:
        stmt = select(AgentEvent).order_by(AgentEvent.created_at.desc(), AgentEvent.id.desc())
        if since is not None:
            stmt = stmt.where(AgentEvent.created_at >= since)
        return self.session.scalars(stmt.limit(limit)).all()

    def latest_of_type(self, event_type: str) -> AgentEvent | None:
        """Most recent event of one type, for 'when did the last cycle run'."""
        return self.session.scalar(
            select(AgentEvent)
            .where(AgentEvent.event_type == event_type)
            .order_by(AgentEvent.created_at.desc(), AgentEvent.id.desc())
        )

    def count_by_type(self, event_type: str) -> int:
        return len(
            self.session.scalars(
                select(AgentEvent).where(AgentEvent.event_type == event_type)
            ).all()
        )
