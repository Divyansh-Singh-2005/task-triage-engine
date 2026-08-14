"""Tests for the decision engine, workload capacity and the evaluation service."""

from __future__ import annotations

import pytest

from agent.decision_engine import decide
from agent.workload_manager import WorkloadStatus
from core.enums import Decision, MonitoringMode, TaskStatus
from core.evaluation import TaskEvaluation
from core.exceptions import EvaluationError
from core.profile import SkillProfile
from core.schemas import AgentConfig, TaskRules
from database.database import create_db_engine, create_session_factory, init_db
from database.repositories import EventRepository, ProjectRepository, TaskRepository
from integrations.llm.base import LLMProvider
from services.evaluation_service import EvaluationService

RULES = TaskRules(minimum_ai_score=80, review_threshold=60, maximum_active_tasks=2)
PROFILE = SkillProfile(skills=["Python"])


def evaluation(score: int, **kw) -> TaskEvaluation:
    payload = {"score": score, "skill_match": score}
    payload.update(kw)
    return TaskEvaluation(**payload)


def free() -> WorkloadStatus:
    return WorkloadStatus(active=0, maximum=2)


def full() -> WorkloadStatus:
    return WorkloadStatus(active=2, maximum=2)


# ------------------------------------------------------------- score bands


def test_low_score_is_skipped():
    outcome = decide(evaluation(40), RULES, free())
    assert outcome.decision is Decision.SKIP
    assert outcome.next_status is TaskStatus.SKIPPED


def test_middle_score_needs_review():
    outcome = decide(evaluation(70), RULES, free())
    assert outcome.decision is Decision.REVIEW_REQUIRED
    assert outcome.next_status is TaskStatus.REVIEW_REQUIRED


def test_high_score_is_a_high_match():
    assert decide(evaluation(95), RULES, free()).decision is Decision.HIGH_MATCH


def test_score_exactly_on_the_bar_is_a_high_match():
    assert decide(evaluation(80), RULES, free()).decision is Decision.HIGH_MATCH


def test_score_exactly_on_the_review_threshold_is_reviewable():
    assert decide(evaluation(60), RULES, free()).decision is Decision.REVIEW_REQUIRED


# --------------------------------------------------------- approval gating


def test_high_match_still_waits_for_a_human_by_default():
    rules = RULES.model_copy(update={"require_human_approval": True})
    outcome = decide(evaluation(95), rules, free())
    assert outcome.decision is Decision.HIGH_MATCH
    assert outcome.next_status is TaskStatus.REVIEW_REQUIRED
    assert outcome.needs_approval


def test_high_match_auto_approves_only_when_approval_is_disabled():
    rules = RULES.model_copy(update={"require_human_approval": False})
    assert decide(evaluation(95), rules, free()).next_status is TaskStatus.APPROVED


# ------------------------------------------------------------- constraints


def test_full_workload_blocks_auto_approval():
    rules = RULES.model_copy(update={"require_human_approval": False})
    outcome = decide(evaluation(95), rules, full())
    assert outcome.next_status is TaskStatus.REVIEW_REQUIRED
    assert outcome.downgraded


def test_passed_deadline_forces_a_skip_despite_a_perfect_score():
    outcome = decide(evaluation(100, deadline_feasibility=0), RULES, free())
    assert outcome.decision is Decision.SKIP
    assert outcome.downgraded


def test_tight_deadline_caps_at_review():
    rules = RULES.model_copy(update={"require_human_approval": False})
    outcome = decide(evaluation(95, deadline_feasibility=20), rules, free())
    assert outcome.decision is Decision.REVIEW_REQUIRED


def test_engine_never_raises_a_decision():
    """Constraints may only lower the outcome, never lift it."""
    outcome = decide(evaluation(30, deadline_feasibility=100), RULES, free())
    assert outcome.decision is Decision.SKIP
    assert outcome.downgraded is False


def test_reasons_are_always_recorded():
    outcome = decide(evaluation(95, deadline_feasibility=10), RULES, full())
    assert len(outcome.reasons) >= 3
    assert "deadline" in outcome.summary().lower()


def test_workload_status_arithmetic():
    assert free().remaining == 2 and free().has_capacity
    assert full().remaining == 0 and not full().has_capacity
    assert "at capacity" in full().note()


# ------------------------------------------------------ evaluation service


@pytest.fixture
def session(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as sess:
        yield sess
    engine.dispose()


class FixedProvider(LLMProvider):
    name = "fixed"

    def __init__(self, score: int = 95) -> None:
        self.score = score
        self.capacity_notes: list[str] = []

    def evaluate_task(self, task, profile, *, capacity_note: str = ""):
        self.capacity_notes.append(capacity_note)
        return evaluation(self.score).model_copy(update={"provider": self.name})


class ExplodingProvider(LLMProvider):
    name = "exploding"

    def evaluate_task(self, task, profile, *, capacity_note: str = ""):
        raise EvaluationError("provider unavailable")


def seed(session, count: int = 1):
    from core.schemas import Task

    project = ProjectRepository(session).get_or_create("Project Dynamo")
    repo = TaskRepository(session)
    for index in range(count):
        repo.add(
            Task(task_id=f"T-{index}", project_name="Project Dynamo", title=f"Task {index}"),
            project,
        )
    session.commit()


def config_with(**rule_overrides) -> AgentConfig:
    return AgentConfig(
        monitoring_mode=MonitoringMode.ACTIVE_PROJECT,
        active_project="Project Dynamo",
        task_rules=RULES.model_copy(update=rule_overrides),
    )


def test_service_evaluates_and_stores_a_decision(session):
    seed(session, 2)
    run = EvaluationService(session, FixedProvider(95)).run(config_with(), PROFILE)
    session.commit()

    assert run.evaluated == 2
    assert run.review_required == 2  # approval required by default
    assert run.ok
    assert len(TaskRepository(session).list_by_status(TaskStatus.REVIEW_REQUIRED)) == 2


def test_service_auto_approves_when_configured(session):
    seed(session, 1)
    run = EvaluationService(session, FixedProvider(95)).run(
        config_with(require_human_approval=False), PROFILE
    )
    session.commit()
    assert run.approved == 1


def test_service_skips_low_scores(session):
    seed(session, 1)
    run = EvaluationService(session, FixedProvider(10)).run(config_with(), PROFILE)
    session.commit()
    assert run.skipped == 1
    assert len(TaskRepository(session).list_by_status(TaskStatus.SKIPPED)) == 1


def test_service_writes_an_evaluation_row(session):
    seed(session, 1)
    EvaluationService(session, FixedProvider(95)).run(config_with(), PROFILE)
    session.commit()

    record = TaskRepository(session).list_by_status()[0]
    assert record.latest_evaluation is not None
    assert record.latest_evaluation.score == 95


def test_provider_failure_marks_the_task_failed_not_the_batch(session):
    seed(session, 2)
    run = EvaluationService(session, ExplodingProvider()).run(config_with(), PROFILE)
    session.commit()

    assert run.failed == 2
    assert run.ok is False
    assert len(TaskRepository(session).list_by_status(TaskStatus.FAILED)) == 2


def test_workload_note_reaches_the_provider(session):
    seed(session, 1)
    provider = FixedProvider(95)
    EvaluationService(session, provider).run(config_with(), PROFILE)
    session.commit()
    assert "active tasks" in provider.capacity_notes[0]


def test_service_records_an_approval_request(session):
    seed(session, 1)
    EvaluationService(session, FixedProvider(95)).run(config_with(), PROFILE)
    session.commit()

    events = EventRepository(session)
    assert events.count_by_type("task_evaluated") == 1
    assert events.count_by_type("approval_requested") == 1
    assert events.count_by_type("evaluation_batch_completed") == 1


def test_already_evaluated_tasks_are_not_re_evaluated(session):
    seed(session, 1)
    service = EvaluationService(session, FixedProvider(95))
    service.run(config_with(), PROFILE)
    session.commit()

    second = service.run(config_with(), PROFILE)
    session.commit()
    assert second.evaluated == 0
