"""Tests for provider selection, the local/hosted providers, and maintenance."""

from __future__ import annotations

import csv
import io

import pytest

from config.settings import Settings
from core.enums import TaskStatus
from core.exceptions import ConfigError, EvaluationError
from core.profile import SkillProfile
from core.schemas import Task
from database.database import create_db_engine, create_session_factory, init_db
from database.repositories import EvaluationRepository, EventRepository, ProjectRepository, TaskRepository
from integrations.llm.factory import KNOWN_PROVIDERS, get_provider
from integrations.llm.heuristic import HeuristicProvider
from integrations.llm.ollama_provider import OllamaProvider
from services.maintenance import REQUEUABLE, MaintenanceService

PROFILE = SkillProfile(skills=["Python"])
GOOD_JSON = '{"score": 82, "skill_match": 80, "estimated_hours": 5, "reason": "fits"}'


def make_task(task_id: str = "T-1") -> Task:
    return Task(task_id=task_id, project_name="Project Dynamo", title="A task")


# ------------------------------------------------------------------ factory


def test_heuristic_is_selected_explicitly():
    provider = get_provider(Settings(llm_provider="heuristic"))
    assert provider.name == "heuristic"


@pytest.mark.parametrize("name", ["anthropic", "openai"])
def test_hosted_providers_fall_back_without_a_key(name):
    """A missing key degrades scoring; it must not crash the loop."""
    settings = Settings(llm_provider=name, anthropic_api_key="", openai_api_key="")
    assert get_provider(settings).name == "heuristic"


def test_anthropic_is_built_when_a_key_exists():
    provider = get_provider(Settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert provider.name == "anthropic"


def test_openai_is_built_when_a_key_exists():
    provider = get_provider(Settings(llm_provider="openai", openai_api_key="k",
                                     openai_model="gpt-4o-mini"))
    assert provider.name == "openai"
    assert provider.model == "gpt-4o-mini"


def test_ollama_needs_no_key():
    """No credential exists to be missing, so there is nothing to fall back from."""
    provider = get_provider(Settings(llm_provider="ollama", ollama_model="llama3.1:8b"))
    assert provider.name == "ollama"
    assert provider.model == "llama3.1:8b"


def test_provider_name_is_case_and_space_insensitive():
    assert get_provider(Settings(llm_provider="  OpenAI  ", openai_api_key="k")).name == "openai"


def test_unknown_provider_is_rejected():
    with pytest.raises(ConfigError):
        get_provider(Settings(llm_provider="mystery-model"))


def test_every_known_provider_resolves():
    for name in KNOWN_PROVIDERS:
        settings = Settings(llm_provider=name, anthropic_api_key="k", openai_api_key="k")
        assert get_provider(settings) is not None


# ------------------------------------------------------------------- ollama


class FakeOllama(OllamaProvider):
    """Substitutes the HTTP layer, leaving the request shaping under test."""

    def __init__(self, response: dict, **kw):
        super().__init__(model="test-model", **kw)
        self.response = response
        self.payloads: list[dict] = []

    def _post(self, path: str, payload: dict) -> dict:
        self.payloads.append({"path": path, **payload})
        return self.response


def test_ollama_evaluates_from_a_chat_response():
    provider = FakeOllama({"message": {"content": GOOD_JSON}})
    evaluation = provider.evaluate_task(make_task(), PROFILE)
    assert evaluation.score == 82
    assert evaluation.provider == "ollama"
    assert evaluation.model == "test-model"


def test_ollama_asks_for_json_and_disables_streaming():
    provider = FakeOllama({"message": {"content": GOOD_JSON}})
    provider.evaluate_task(make_task(), PROFILE)
    sent = provider.payloads[0]
    assert sent["path"] == "/api/chat"
    assert sent["format"] == "json"
    assert sent["stream"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_ollama_surfaces_an_error_field():
    provider = FakeOllama({"error": "model not found"})
    with pytest.raises(EvaluationError, match="model not found"):
        provider.evaluate_task(make_task(), PROFILE)


def test_ollama_rejects_an_empty_message():
    provider = FakeOllama({"message": {"content": ""}})
    with pytest.raises(EvaluationError):
        provider.evaluate_task(make_task(), PROFILE)


def test_ollama_base_url_loses_its_trailing_slash():
    assert OllamaProvider("m", base_url="http://host:11434/").base_url == "http://host:11434"


# -------------------------------------------------------------- maintenance


@pytest.fixture
def session(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as sess:
        yield sess
    engine.dispose()


def seed(session, status: TaskStatus, count: int = 1, project: str = "Project Dynamo"):
    repo = TaskRepository(session)
    proj = ProjectRepository(session).get_or_create(project)
    offset = len(repo.list_by_status())
    for index in range(offset, offset + count):
        record = repo.add(
            Task(task_id=f"T-{index}", project_name=project, title=f"Task {index}"), proj
        )
        repo.set_status(record, status)
    session.commit()


def test_requeue_moves_tasks_back_to_discovered(session):
    seed(session, TaskStatus.SKIPPED, 2)
    seed(session, TaskStatus.FAILED, 1)
    result = MaintenanceService(session).requeue()
    session.commit()

    assert result.requeued == 3
    assert len(TaskRepository(session).list_by_status(TaskStatus.DISCOVERED)) == 3


def test_requeue_leaves_approved_work_alone(session):
    seed(session, TaskStatus.APPROVED, 2)
    seed(session, TaskStatus.SKIPPED, 1)
    MaintenanceService(session).requeue()
    session.commit()
    assert len(TaskRepository(session).list_by_status(TaskStatus.APPROVED)) == 2


def test_requeue_can_target_one_status(session):
    seed(session, TaskStatus.SKIPPED, 2)
    seed(session, TaskStatus.REVIEW_REQUIRED, 1)
    result = MaintenanceService(session).requeue(TaskStatus.SKIPPED)
    session.commit()
    assert result.requeued == 2
    assert len(TaskRepository(session).list_by_status(TaskStatus.REVIEW_REQUIRED)) == 1


def test_requeue_refuses_a_disallowed_status(session):
    with pytest.raises(ValueError):
        MaintenanceService(session).requeue(TaskStatus.APPROVED)


def test_requeue_keeps_the_old_evaluations(session):
    from core.enums import Decision

    seed(session, TaskStatus.SKIPPED, 1)
    record = TaskRepository(session).list_by_status()[0]
    EvaluationRepository(session).add(record, score=40, decision=Decision.SKIP)
    session.commit()

    MaintenanceService(session).requeue()
    session.commit()
    assert len(TaskRepository(session).list_by_status()[0].evaluations) == 1


def test_requeue_is_recorded(session):
    seed(session, TaskStatus.SKIPPED, 1)
    MaintenanceService(session).requeue()
    session.commit()
    assert EventRepository(session).count_by_type("tasks_requeued") == 1


def test_requeue_of_nothing_logs_nothing(session):
    result = MaintenanceService(session).requeue()
    session.commit()
    assert result.requeued == 0
    assert EventRepository(session).count_by_type("tasks_requeued") == 0


def test_requeue_counts_cover_every_requeueable_status(session):
    seed(session, TaskStatus.SKIPPED, 2)
    counts = MaintenanceService(session).requeue_counts()
    assert set(counts) == set(REQUEUABLE)
    assert counts[TaskStatus.SKIPPED] == 2


# --------------------------------------------------------------- reconcile


@pytest.fixture
def manager(tmp_path):
    import json

    from agent.project_manager import ProjectManager

    defaults = tmp_path / "defaults.json"
    defaults.write_text(json.dumps({
        "monitoring_mode": "ACTIVE_PROJECT",
        "active_project": "Project Dynamo",
        "known_projects": ["Project Dynamo"],
    }), encoding="utf-8")
    return ProjectManager(tmp_path / "agent_config.json", defaults)


def test_reconcile_adds_database_projects_to_config(session, manager):
    ProjectRepository(session).get_or_create("Project Gamma")
    session.commit()

    result = MaintenanceService(session).reconcile_projects(manager)
    session.commit()
    assert result.added_to_config == ["Project Gamma"]
    assert "Project Gamma" in manager.load(force=True).known_projects


def test_reconcile_adds_config_projects_to_the_database(session, manager):
    manager.register_project("Project Beta")
    result = MaintenanceService(session).reconcile_projects(manager)
    session.commit()
    assert "Project Beta" in result.added_to_database
    assert ProjectRepository(session).get_by_name("Project Beta") is not None


def test_reconcile_is_idempotent(session, manager):
    manager.register_project("Project Beta")
    service = MaintenanceService(session)
    service.reconcile_projects(manager)
    session.commit()
    second = service.reconcile_projects(manager)
    session.commit()
    assert second.changed is False


def test_reconcile_matches_case_insensitively(session, manager):
    ProjectRepository(session).get_or_create("project  dynamo")
    session.commit()
    result = MaintenanceService(session).reconcile_projects(manager)
    assert result.added_to_config == []


# ------------------------------------------------------------------ export


def test_task_export_has_a_header_and_one_row_per_task(session):
    seed(session, TaskStatus.REVIEW_REQUIRED, 2)
    rows = list(csv.reader(io.StringIO(MaintenanceService(session).export_tasks_csv())))
    assert rows[0][0] == "task_id"
    assert len(rows) == 3


def test_task_export_flattens_the_latest_evaluation(session):
    from core.enums import Decision

    seed(session, TaskStatus.REVIEW_REQUIRED, 1)
    record = TaskRepository(session).list_by_status()[0]
    repo = EvaluationRepository(session)
    repo.add(record, score=40, decision=Decision.SKIP)
    repo.add(record, score=88, decision=Decision.HIGH_MATCH, risks=["scope", "clarity"])
    session.commit()

    rows = list(csv.DictReader(io.StringIO(MaintenanceService(session).export_tasks_csv())))
    assert rows[0]["score"] == "88"
    assert rows[0]["risks"] == "scope | clarity"


def test_evaluation_export_keeps_the_whole_history(session):
    from core.enums import Decision

    seed(session, TaskStatus.REVIEW_REQUIRED, 1)
    record = TaskRepository(session).list_by_status()[0]
    repo = EvaluationRepository(session)
    repo.add(record, score=40, decision=Decision.SKIP)
    repo.add(record, score=88, decision=Decision.HIGH_MATCH)
    session.commit()

    rows = list(csv.DictReader(io.StringIO(MaintenanceService(session).export_evaluations_csv())))
    assert [r["score"] for r in rows] == ["40", "88"]


def test_export_of_an_empty_database_is_just_a_header(session):
    rows = list(csv.reader(io.StringIO(MaintenanceService(session).export_tasks_csv())))
    assert len(rows) == 1


def test_export_survives_a_task_with_no_evaluation(session):
    seed(session, TaskStatus.DISCOVERED, 1)
    rows = list(csv.DictReader(io.StringIO(MaintenanceService(session).export_tasks_csv())))
    assert rows[0]["score"] == ""
