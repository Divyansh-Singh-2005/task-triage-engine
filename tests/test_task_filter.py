"""Tests for the project filter.

Every monitoring mode gets the same three cases: the in-scope project, an
out-of-scope project, and a name that differs only by casing/whitespace.
"""

from __future__ import annotations

import pytest

from agent.task_filter import evaluate_project_filter, filter_tasks, should_process_task
from core.enums import MonitoringMode
from core.schemas import AgentConfig, Task


def make_task(project: str, task_id: str = "T-1") -> Task:
    return Task(task_id=task_id, project_name=project, title=f"Task {task_id}")


# ------------------------------------------------------------- ACTIVE_PROJECT


def test_active_project_accepts_matching_project():
    config = AgentConfig(monitoring_mode=MonitoringMode.ACTIVE_PROJECT, active_project="Project Dynamo")
    assert should_process_task(make_task("Project Dynamo"), config) is True


def test_active_project_rejects_other_project():
    config = AgentConfig(monitoring_mode=MonitoringMode.ACTIVE_PROJECT, active_project="Project Dynamo")
    assert should_process_task(make_task("Project Alpha"), config) is False


@pytest.mark.parametrize("variant", ["project dynamo", "PROJECT DYNAMO", "  Project   Dynamo  "])
def test_active_project_is_case_and_whitespace_insensitive(variant):
    config = AgentConfig(monitoring_mode=MonitoringMode.ACTIVE_PROJECT, active_project="Project Dynamo")
    assert should_process_task(make_task(variant), config) is True


# ---------------------------------------------------------- SELECTED_PROJECTS


def test_selected_projects_accepts_member():
    config = AgentConfig(
        monitoring_mode=MonitoringMode.SELECTED_PROJECTS,
        selected_projects=["Project Dynamo", "Project Gamma"],
    )
    assert should_process_task(make_task("Project Gamma"), config) is True


def test_selected_projects_rejects_non_member():
    config = AgentConfig(
        monitoring_mode=MonitoringMode.SELECTED_PROJECTS,
        selected_projects=["Project Dynamo", "Project Gamma"],
    )
    assert should_process_task(make_task("Project Beta"), config) is False


def test_selected_projects_ignores_active_project():
    """A project being 'active' must not leak scope into SELECTED_PROJECTS mode."""
    config = AgentConfig(
        monitoring_mode=MonitoringMode.SELECTED_PROJECTS,
        active_project="Project Alpha",
        selected_projects=["Project Dynamo"],
    )
    assert should_process_task(make_task("Project Alpha"), config) is False


# ----------------------------------------------------------------- ANY_PROJECT


@pytest.mark.parametrize("project", ["Project Dynamo", "Project Zeta", "Something Unheard Of"])
def test_any_project_accepts_everything(project):
    config = AgentConfig(monitoring_mode=MonitoringMode.ANY_PROJECT)
    assert should_process_task(make_task(project), config) is True


def test_any_project_does_not_need_an_active_project():
    config = AgentConfig(monitoring_mode=MonitoringMode.ANY_PROJECT, active_project=None)
    assert should_process_task(make_task("Anything"), config) is True


# --------------------------------------------------------------- explanations


def test_decision_carries_a_reason():
    config = AgentConfig(monitoring_mode=MonitoringMode.ACTIVE_PROJECT, active_project="Project Dynamo")
    decision = evaluate_project_filter(make_task("Project Alpha"), config)
    assert decision.process is False
    assert "Project Alpha" in decision.reason


def test_filter_tasks_preserves_order():
    config = AgentConfig(
        monitoring_mode=MonitoringMode.SELECTED_PROJECTS,
        selected_projects=["Project Dynamo"],
    )
    tasks = [
        make_task("Project Dynamo", "T-1"),
        make_task("Project Alpha", "T-2"),
        make_task("Project Dynamo", "T-3"),
    ]
    assert [t.task_id for t in filter_tasks(tasks, config)] == ["T-1", "T-3"]
