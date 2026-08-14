"""Task ingestion: source -> project filter -> duplicate guard -> database.

The whole pipeline runs inside one transaction per ingest call, so a failure
partway through leaves no half-ingested batch behind. Individual bad tasks are
caught and recorded rather than aborting the run - one malformed listing must
not stop the other nine from being seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from agent.sources.base import TaskSource
from agent.task_filter import evaluate_project_filter
from core.exceptions import TaskSourceError
from core.schemas import AgentConfig, Task
from database.repositories import EventRepository, ProjectRepository, TaskRepository


@dataclass
class IngestionReport:
    """What one ingest run actually did, for logs and the dashboard."""

    source: str
    fetched: int = 0
    stored: int = 0
    duplicates: int = 0
    filtered_out: int = 0
    errors: list[str] = field(default_factory=list)
    stored_task_ids: list[int] = field(default_factory=list)
    filter_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"{self.source}: fetched {self.fetched}, stored {self.stored}, "
            f"duplicates {self.duplicates}, filtered out {self.filtered_out}, "
            f"errors {len(self.errors)}"
        )


class TaskIngestionService:
    """Pulls from a :class:`TaskSource` and persists what is in scope."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.projects = ProjectRepository(session)
        self.tasks = TaskRepository(session)
        self.events = EventRepository(session)

    def ingest(self, source: TaskSource, config: AgentConfig) -> IngestionReport:
        report = IngestionReport(source=source.name)

        try:
            fetched = source.fetch()
        except TaskSourceError as exc:
            report.errors.append(str(exc))
            self.events.log(
                "source_failed", message=str(exc), payload={"source": source.name}
            )
            return report

        report.fetched = len(fetched)

        for task in fetched:
            try:
                self._ingest_one(task, config, report)
            except Exception as exc:  # one bad task must not kill the batch
                report.errors.append(f"{task.task_id}: {exc}")
                self.events.log(
                    "task_ingest_failed",
                    message=str(exc),
                    project_name=task.project_name,
                    payload={"task_id": task.task_id, "source": source.name},
                )

        self.events.log(
            "ingest_completed",
            message=report.summary(),
            payload={
                "source": source.name,
                "fetched": report.fetched,
                "stored": report.stored,
                "duplicates": report.duplicates,
                "filtered_out": report.filtered_out,
                "errors": len(report.errors),
            },
        )
        return report

    def _ingest_one(self, task: Task, config: AgentConfig, report: IngestionReport) -> None:
        # Duplicate check comes first: a task already stored should not be
        # re-logged as filtered just because the mode changed since.
        existing = self.tasks.find_duplicate(task)
        if existing is not None:
            report.duplicates += 1
            self.events.log(
                "duplicate_task_ignored",
                message=f"{task.task_id} already stored as row {existing.id}",
                task_id=existing.id,
                project_name=task.project_name,
                payload={"task_id": task.task_id},
            )
            return

        decision = evaluate_project_filter(task, config)
        report.filter_reasons[task.task_id] = decision.reason
        if not decision.process:
            report.filtered_out += 1
            self.events.log(
                "task_filtered",
                message=decision.reason,
                project_name=task.project_name,
                payload={"task_id": task.task_id, "mode": config.monitoring_mode.value},
            )
            return

        project = self.projects.get_or_create(task.project_name)
        record = self.tasks.add(task, project)
        report.stored += 1
        report.stored_task_ids.append(record.id)
        self.events.log(
            "task_discovered",
            message=f"stored {task.title!r}",
            task_id=record.id,
            project_name=project.name,
            payload={"task_id": task.task_id, "source": task.source},
        )
