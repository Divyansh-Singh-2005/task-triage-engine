"""Project filter: decides whether a discovered task is in scope.

This module is deliberately pure - no I/O, no logging side effects - so it can
be tested exhaustively and reused by the dashboard for "why was this ignored?"
explanations. Agent start/stop state is *not* checked here; that belongs to
the orchestrator, which should not run this at all when stopped.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

from core.enums import MonitoringMode
from core.naming import project_key
from core.schemas import AgentConfig, Task


class FilterDecision(NamedTuple):
    """A filter outcome plus a human-readable reason, for logs and the UI."""

    process: bool
    reason: str


def evaluate_project_filter(task: Task, config: AgentConfig) -> FilterDecision:
    """Explain whether ``task`` passes the project filter."""
    mode = config.monitoring_mode

    if mode is MonitoringMode.ANY_PROJECT:
        return FilterDecision(True, "ANY_PROJECT mode: no project restriction")

    task_key = project_key(task.project_name)

    if mode is MonitoringMode.ACTIVE_PROJECT:
        active = config.active_project
        if not active:
            return FilterDecision(False, "ACTIVE_PROJECT mode with no active project set")
        if task_key == project_key(active):
            return FilterDecision(True, f"matches active project {active!r}")
        return FilterDecision(
            False, f"project {task.project_name!r} is not the active project {active!r}"
        )

    if mode is MonitoringMode.SELECTED_PROJECTS:
        selected = {project_key(p): p for p in config.selected_projects}
        if task_key in selected:
            return FilterDecision(True, f"matches selected project {selected[task_key]!r}")
        return FilterDecision(
            False, f"project {task.project_name!r} is not in the selected projects"
        )

    # Unreachable while MonitoringMode stays exhaustive, but fail closed if a
    # new mode is added and this function is not updated.
    return FilterDecision(False, f"unhandled monitoring mode {mode!r}; failing closed")


def should_process_task(task: Task, config: AgentConfig) -> bool:
    """Boolean form of :func:`evaluate_project_filter`."""
    return evaluate_project_filter(task, config).process


def filter_tasks(tasks: Iterable[Task], config: AgentConfig) -> list[Task]:
    """Return only the tasks that pass the project filter, order preserved."""
    return [task for task in tasks if should_process_task(task, config)]
