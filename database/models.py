"""ORM models.

Naming note: the Pydantic ``Task`` in ``core.schemas`` is the in-flight value
object; ``TaskRecord`` here is its persisted form. Keeping the names distinct
makes it obvious at every call site which one is in hand.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.enums import Complexity, Decision, TaskStatus
from database.database import Base, UtcDateTime, utcnow


def _enum(enum_cls) -> SAEnum:
    """Store enums by value as plain strings, so the DB stays readable."""
    return SAEnum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda e: [member.value for member in e],
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # Canonical lowercase form; the unique constraint lives here so that
    # "Project Dynamo" and "project dynamo" can never become two rows.
    key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)

    tasks: Mapped[list["TaskRecord"]] = relationship(back_populates="project")

    def __repr__(self) -> str:
        return f"<Project {self.name!r}>"


class TaskRecord(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        # Primary duplicate guard: a source may not report the same id twice.
        UniqueConstraint("source", "external_task_id", name="uq_task_source_external_id"),
        Index("ix_task_fingerprint", "fingerprint"),
        Index("ix_task_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="manual")
    external_task_id: Mapped[str] = mapped_column(String(200), nullable=False)
    # Secondary duplicate guard for sources with no stable id of their own.
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    requirements: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    deadline: Mapped[datetime | None] = mapped_column(UtcDateTime, nullable=True)
    reward: Mapped[str | None] = mapped_column(String(200), nullable=True)
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    status: Mapped[TaskStatus] = mapped_column(
        _enum(TaskStatus), default=TaskStatus.DISCOVERED, nullable=False
    )
    discovered_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    project: Mapped[Project] = relationship(back_populates="tasks")
    evaluations: Mapped[list["Evaluation"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="Evaluation.created_at"
    )

    @property
    def latest_evaluation(self) -> "Evaluation | None":
        return self.evaluations[-1] if self.evaluations else None

    def __repr__(self) -> str:
        return f"<TaskRecord {self.external_task_id!r} {self.status.value}>"


class Evaluation(Base):
    """One AI evaluation of one task.

    Stored append-only: re-evaluating a task adds a row rather than mutating
    the old one, so a scoring-prompt change is auditable after the fact.
    """

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    score: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[Decision] = mapped_column(_enum(Decision), nullable=False)
    estimated_hours: Mapped[float | None] = mapped_column(nullable=True)
    skill_match: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deadline_feasibility: Mapped[int | None] = mapped_column(Integer, nullable=True)
    complexity: Mapped[Complexity | None] = mapped_column(_enum(Complexity), nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    risks: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)

    task: Mapped[TaskRecord] = relationship(back_populates="evaluations")


class AgentEvent(Base):
    """Append-only audit trail.

    Never write credentials or tokens into ``payload``; this table is dumped
    wholesale into the dashboard.
    """

    __tablename__ = "agent_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )
    project_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<AgentEvent {self.event_type}>"
