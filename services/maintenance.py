"""Maintenance operations: bulk requeue, project reconciliation, export.

These are the operations you reach for when something *outside* the pipeline
changed - you edited the skill profile, you registered a project in the UI, you
want the numbers in a spreadsheet. None of them belong in the ingest/evaluate
path, and none should be reachable by accident.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from agent.project_manager import ProjectManager
from core.enums import TaskStatus
from core.naming import project_key
from database.repositories import EventRepository, ProjectRepository, TaskRepository

#: Statuses safe to requeue in bulk. Approved and in-flight work is excluded:
#: re-scoring something you already committed to is never what you meant.
REQUEUABLE = (TaskStatus.SKIPPED, TaskStatus.FAILED, TaskStatus.REVIEW_REQUIRED)


@dataclass
class RequeueResult:
    requeued: int = 0
    skipped_statuses: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.requeued:
            return "nothing to requeue"
        return f"requeued {self.requeued} task(s) for re-evaluation"


@dataclass
class ReconcileResult:
    """What drifted between the config file and the projects table."""

    added_to_config: list[str] = field(default_factory=list)
    added_to_database: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added_to_config or self.added_to_database)

    def summary(self) -> str:
        if not self.changed:
            return "projects already in sync"
        parts = []
        if self.added_to_config:
            parts.append(f"config gained {', '.join(self.added_to_config)}")
        if self.added_to_database:
            parts.append(f"database gained {', '.join(self.added_to_database)}")
        return "; ".join(parts)


class MaintenanceService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.tasks = TaskRepository(session)
        self.projects = ProjectRepository(session)
        self.events = EventRepository(session)

    # ------------------------------------------------------------- requeue

    def requeue(self, *statuses: TaskStatus) -> RequeueResult:
        """Send tasks back to DISCOVERED so the next cycle re-scores them.

        Previous evaluation rows are left alone - the table is append-only, so
        after a profile change you can still see what the old scoring said.
        """
        wanted = tuple(statuses) or REQUEUABLE
        refused = [s for s in wanted if s not in REQUEUABLE]
        if refused:
            raise ValueError(
                f"cannot requeue from {', '.join(s.value for s in refused)}; "
                f"allowed: {', '.join(s.value for s in REQUEUABLE)}"
            )

        result = RequeueResult()
        for record in list(self.tasks.list_by_status(*wanted)):
            self.tasks.set_status(record, TaskStatus.DISCOVERED)
            result.requeued += 1

        if result.requeued:
            self.events.log(
                "tasks_requeued",
                message=result.summary(),
                payload={"count": result.requeued,
                         "from": [s.value for s in wanted]},
            )
        return result

    def requeue_counts(self) -> dict[TaskStatus, int]:
        """How many tasks sit in each requeueable status, for the UI."""
        return {status: len(self.tasks.list_by_status(status)) for status in REQUEUABLE}

    # ----------------------------------------------------------- reconcile

    def reconcile_projects(self, manager: ProjectManager) -> ReconcileResult:
        """Make the config known projects and the projects table agree.

        They are populated independently: registering a project in the sidebar
        only touches config, while ingesting a task only touches the database.
        Without this the sidebar dropdown and the board filter disagree.
        """
        result = ReconcileResult()
        config = manager.load(force=True)

        db_names = [p.name for p in self.projects.list_all()]
        db_keys = {project_key(name) for name in db_names}
        config_keys = {project_key(name) for name in config.known_projects}

        for name in db_names:
            if project_key(name) not in config_keys:
                manager.register_project(name)
                result.added_to_config.append(name)

        for name in config.known_projects:
            if project_key(name) not in db_keys:
                self.projects.get_or_create(name)
                result.added_to_database.append(name)

        if result.changed:
            self.events.log("projects_reconciled", message=result.summary())
        return result

    # -------------------------------------------------------------- export

    def export_tasks_csv(self) -> str:
        """One row per task, with its latest evaluation flattened in."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([
            "task_id", "external_id", "source", "project", "title", "status",
            "score", "decision", "skill_match", "deadline_feasibility",
            "estimated_hours", "complexity", "risks", "reason",
            "deadline", "reward", "url", "discovered_at", "updated_at",
        ])
        for record in self.tasks.list_by_status():
            latest = record.latest_evaluation
            writer.writerow([
                record.id, record.external_task_id, record.source,
                record.project.name, record.title, record.status.value,
                latest.score if latest else "",
                latest.decision.value if latest else "",
                latest.skill_match if latest else "",
                latest.deadline_feasibility if latest else "",
                latest.estimated_hours if latest else "",
                latest.complexity.value if latest and latest.complexity else "",
                " | ".join(latest.risks or []) if latest else "",
                latest.reason if latest else "",
                record.deadline.isoformat() if record.deadline else "",
                record.reward or "", record.url or "",
                record.discovered_at.isoformat() if record.discovered_at else "",
                record.updated_at.isoformat() if record.updated_at else "",
            ])
        return buffer.getvalue()

    def export_evaluations_csv(self) -> str:
        """Every evaluation, not just the latest - this is the scoring history."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([
            "evaluation_id", "task_id", "external_id", "project", "title",
            "score", "decision", "skill_match", "deadline_feasibility",
            "estimated_hours", "complexity", "model", "risks", "reason", "created_at",
        ])
        for record in self.tasks.list_by_status():
            for evaluation in record.evaluations:
                writer.writerow([
                    evaluation.id, record.id, record.external_task_id,
                    record.project.name, record.title,
                    evaluation.score, evaluation.decision.value,
                    evaluation.skill_match or "", evaluation.deadline_feasibility or "",
                    evaluation.estimated_hours or "",
                    evaluation.complexity.value if evaluation.complexity else "",
                    evaluation.model or "",
                    " | ".join(evaluation.risks or []),
                    evaluation.reason,
                    evaluation.created_at.isoformat() if evaluation.created_at else "",
                ])
        return buffer.getvalue()
