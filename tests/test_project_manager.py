"""Tests for configuration loading, validation, mutation and persistence."""

from __future__ import annotations

import json

import pytest

from agent.project_manager import ProjectManager
from core.enums import AgentStatus, MonitoringMode
from core.exceptions import ConfigError, UnknownProjectError
from core.schemas import AgentConfig


@pytest.fixture
def manager(tmp_path):
    defaults = tmp_path / "defaults.json"
    defaults.write_text(
        json.dumps(
            {
                "agent_status": "stopped",
                "monitoring_mode": "ACTIVE_PROJECT",
                "active_project": "Project Dynamo",
                "selected_projects": ["Project Dynamo"],
                "known_projects": ["Project Dynamo"],
                "task_rules": {
                    "minimum_ai_score": 80,
                    "review_threshold": 75,
                    "maximum_active_tasks": 2,
                    "require_human_approval": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return ProjectManager(tmp_path / "agent_config.json", defaults)


def test_falls_back_to_defaults_when_no_config_exists(manager):
    config = manager.load()
    assert config.active_project == "Project Dynamo"
    assert config.agent_status is AgentStatus.STOPPED


def test_changes_persist_to_disk(manager):
    manager.set_active_project("Project Alpha")
    reloaded = ProjectManager(manager.config_path).load()
    assert reloaded.active_project == "Project Alpha"


def test_setting_active_project_registers_it(manager):
    config = manager.set_active_project("Project Gamma")
    assert "Project Gamma" in config.known_projects


def test_strict_mode_rejects_unregistered_project(manager):
    with pytest.raises(UnknownProjectError):
        manager.set_active_project("Project Nowhere", auto_register=False)


def test_add_and_remove_selected_projects(manager):
    manager.add_selected_project("Project Alpha")
    assert set(manager.config.selected_projects) == {"Project Dynamo", "Project Alpha"}
    manager.remove_selected_project("project alpha")  # case-insensitive
    assert manager.config.selected_projects == ["Project Dynamo"]


def test_selected_projects_are_deduplicated(manager):
    manager.add_selected_project("Project Dynamo")
    manager.add_selected_project("  project   dynamo  ")
    assert manager.config.selected_projects == ["Project Dynamo"]


def test_switching_modes(manager):
    config = manager.set_monitoring_mode(MonitoringMode.ANY_PROJECT)
    assert config.monitoring_mode is MonitoringMode.ANY_PROJECT
    config = manager.set_monitoring_mode("SELECTED_PROJECTS")
    assert config.monitoring_mode is MonitoringMode.SELECTED_PROJECTS


def test_unknown_mode_is_rejected(manager):
    with pytest.raises(ConfigError):
        manager.set_monitoring_mode("EVERY_PROJECT_EVER")


def test_selected_mode_requires_at_least_one_project(manager):
    manager.set_monitoring_mode("SELECTED_PROJECTS")
    with pytest.raises(ConfigError):
        manager.remove_selected_project("Project Dynamo")


def test_rejected_change_leaves_config_untouched(manager):
    before = manager.config.model_dump()
    with pytest.raises(ConfigError):
        manager.set_task_rules(minimum_ai_score=500)
    assert manager.config.model_dump() == before


def test_task_rule_threshold_ordering_is_enforced(manager):
    with pytest.raises(ConfigError):
        manager.set_task_rules(minimum_ai_score=60, review_threshold=90)


def test_task_rules_update_partially(manager):
    config = manager.set_task_rules(maximum_active_tasks=5)
    assert config.task_rules.maximum_active_tasks == 5
    assert config.task_rules.minimum_ai_score == 80


def test_agent_status_toggle(manager):
    assert manager.set_agent_status("running").agent_status is AgentStatus.RUNNING


def test_corrupt_config_raises_config_error(tmp_path):
    bad = tmp_path / "agent_config.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        ProjectManager(bad).load()


def test_active_project_mode_requires_an_active_project():
    with pytest.raises(Exception):
        AgentConfig(monitoring_mode=MonitoringMode.ACTIVE_PROJECT, active_project=None)


def test_saved_file_is_valid_json_with_lf_endings(manager):
    manager.set_agent_status("running")
    raw = manager.config_path.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8"))["agent_status"] == "running"
