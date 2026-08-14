"""Validated data models.

These are the contracts the rest of the application programs against. Nothing
downstream should accept a raw dict where one of these types is expected.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.enums import AgentStatus, MonitoringMode, TaskStatus
from core.naming import dedupe_projects, normalize_project_name


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskRules(BaseModel):
    """Deterministic rules layered on top of the AI evaluation.

    ``review_threshold`` is the floor for human review; ``minimum_ai_score``
    is the floor for an automatic high-match. Scores below the review
    threshold are skipped outright.
    """

    model_config = ConfigDict(extra="forbid")

    minimum_ai_score: int = Field(default=80, ge=0, le=100)
    review_threshold: int = Field(default=75, ge=0, le=100)
    maximum_active_tasks: int = Field(default=2, ge=0)
    require_human_approval: bool = True

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "TaskRules":
        if self.review_threshold > self.minimum_ai_score:
            raise ValueError(
                f"review_threshold ({self.review_threshold}) cannot exceed "
                f"minimum_ai_score ({self.minimum_ai_score})"
            )
        return self


class AgentConfig(BaseModel):
    """The full runtime configuration, persisted as JSON.

    Invariants enforced here rather than at the call site:
      * project names are normalized and de-duplicated
      * ``known_projects`` is always a superset of active + selected
      * the mode in force has the data it needs to be meaningful
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    agent_status: AgentStatus = AgentStatus.STOPPED
    monitoring_mode: MonitoringMode = MonitoringMode.ACTIVE_PROJECT
    active_project: str | None = None
    selected_projects: list[str] = Field(default_factory=list)
    known_projects: list[str] = Field(default_factory=list)
    task_rules: TaskRules = Field(default_factory=TaskRules)

    @field_validator("active_project", mode="before")
    @classmethod
    def _normalize_active(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return normalize_project_name(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _reconcile(self) -> "AgentConfig":
        # Bypass validate_assignment while we normalize, to avoid recursion.
        selected = dedupe_projects(self.selected_projects)
        known = dedupe_projects(
            self.known_projects
            + ([self.active_project] if self.active_project else [])
            + selected
        )
        object.__setattr__(self, "selected_projects", selected)
        object.__setattr__(self, "known_projects", known)

        if self.monitoring_mode is MonitoringMode.ACTIVE_PROJECT and not self.active_project:
            raise ValueError("ACTIVE_PROJECT mode requires active_project to be set")
        if self.monitoring_mode is MonitoringMode.SELECTED_PROJECTS and not selected:
            raise ValueError("SELECTED_PROJECTS mode requires at least one selected project")
        return self


class Task(BaseModel):
    """A task as this application understands it.

    ``task_id`` is the identifier assigned by whatever source produced the
    task. ``fingerprint`` is a content hash used as a secondary duplicate
    guard for sources that do not expose a stable id.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    project_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    requirements: list[str] = Field(default_factory=list)
    deadline: datetime | None = None
    reward: str | None = None
    url: str | None = None
    source: str = "manual"
    discovered_at: datetime = Field(default_factory=_utcnow)
    status: TaskStatus = TaskStatus.DISCOVERED

    @field_validator("project_name", "title", "task_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @property
    def fingerprint(self) -> str:
        """Stable hash of the identifying fields, for duplicate detection."""
        payload = "\x1f".join([self.source, self.project_name, self.title, self.task_id])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
