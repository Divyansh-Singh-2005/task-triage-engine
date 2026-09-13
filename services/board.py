"""Read model for the dashboard.

ORM objects detach the moment their session closes, and Streamlit reruns
constantly, so the UI is handed plain dataclasses instead. This also keeps the
dashboard from reaching into the database directly and growing its own
queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from agent.workload_manager import WorkloadManager, WorkloadStatus
from core.enums import TaskStatus
from core.schemas import TaskRules
from database.repositories import EventRepository, ProjectRepository, TaskRepository


@dataclass(frozen=True)
class TaskView:
    id: int
    external_id: str
    project: str
    title: str
    status: TaskStatus
    source: str
    score: int | None = None
    decision: str | None = None
    estimated_hours: float | None = None
    complexity: str | None = None
    reason: str = ""
    risks: list[str] = field(default_factory=list)
    deadline: datetime | None = None
    reward: str | None = None
    url: str | None = None
    discovered_at: datetime | None = None

    @property
    def score_text(self) -> str:
        return "-" if self.score is None else str(self.score)


@dataclass(frozen=True)
class EventView:
    event_type: str
    message: str
    project: str | None
    created_at: datetime


@dataclass
class BoardSnapshot:
    tasks: list[TaskView] = field(default_factory=list)
    events: list[EventView] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    workload: WorkloadStatus | None = None
    last_cycle: datetime | None = None
    last_cycle_summary: str = ""

    def by_status(self, *statuses: TaskStatus) -> list[TaskView]:
        wanted = set(statuses)
        return [t for t in self.tasks if t.status in wanted]

    def count(self, status: TaskStatus) -> int:
        return self.counts.get(status.value, 0)


def build_snapshot(session: Session, rules: TaskRules, *, event_limit: int = 25) -> BoardSnapshot:
    """Assemble everything the dashboard renders, in one pass."""
    tasks_repo = TaskRepository(session)
    records = tasks_repo.list_by_status()

    views: list[TaskView] = []
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status.value] = counts.get(record.status.value, 0) + 1
        latest = record.latest_evaluation
        views.append(TaskView(
            id=record.id, external_id=record.external_task_id, project=record.project.name,
            title=record.title, status=record.status, source=record.source,
            score=latest.score if latest else None,
            decision=latest.decision.value if latest else None,
            estimated_hours=latest.estimated_hours if latest else None,
            complexity=latest.complexity.value if latest and latest.complexity else None,
            reason=latest.reason if latest else "",
            risks=list(latest.risks or []) if latest else [],
            deadline=record.deadline, reward=record.reward, url=record.url,
            discovered_at=record.discovered_at,
        ))

    event_repo = EventRepository(session)
    events = [EventView(event_type=e.event_type, message=e.message, project=e.project_name,
                        created_at=e.created_at) for e in event_repo.recent(event_limit)]

    cycle = event_repo.latest_of_type("cycle_completed")

    return BoardSnapshot(
        tasks=views, events=events,
        projects=[p.name for p in ProjectRepository(session).list_all()],
        counts=counts, workload=WorkloadManager(tasks_repo).status(rules),
        last_cycle=cycle.created_at if cycle else None,
        last_cycle_summary=cycle.message if cycle else "",
    )
