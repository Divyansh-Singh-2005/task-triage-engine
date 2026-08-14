"""Enumerations used across the agent.

All of these subclass ``str`` so they serialize cleanly to JSON and compare
equal to their string form, which keeps the config file human-editable.
"""

from __future__ import annotations

from enum import Enum


class MonitoringMode(str, Enum):
    """How the project filter decides which tasks are in scope."""

    ACTIVE_PROJECT = "ACTIVE_PROJECT"
    SELECTED_PROJECTS = "SELECTED_PROJECTS"
    ANY_PROJECT = "ANY_PROJECT"


class AgentStatus(str, Enum):
    """Whether the monitoring loop is running."""

    STOPPED = "stopped"
    RUNNING = "running"


class TaskStatus(str, Enum):
    """Lifecycle of a single task inside this application.

    Note this is the agent's view of the task, not the platform's.
    """

    DISCOVERED = "DISCOVERED"
    EVALUATING = "EVALUATING"
    SKIPPED = "SKIPPED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    CLAIM_PENDING = "CLAIM_PENDING"
    CLAIMED = "CLAIMED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class Decision(str, Enum):
    """Output of the decision engine (Phase 2)."""

    HIGH_MATCH = "HIGH_MATCH"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    SKIP = "SKIP"


class Complexity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
