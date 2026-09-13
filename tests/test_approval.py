"""Tests for the approval workflow and the dashboard read model."""

from __future__ import annotations

import pytest

from core.enums import TaskStatus
from core.schemas import Task, TaskRules
from database.database import create_db_engine, create_session_factory, init_db
from database.repositories import EventRepository, ProjectRepository, TaskRepository
from services.approval_service import ApprovalError, ApprovalService
from services.board import build_snapshot

RULES = TaskRules(minimum_ai_score=80, review_threshold=60, maximum_active_tasks=2)


@pytest.fixture
def session(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as sess:
        yield sess
    engine.dispose()


def seed(session, status: TaskStatus = TaskStatus.REVIEW_REQUIRED, count: int = 1) -> list[int]:
    """Insert tasks in a given status.

    Ids are offset by what is already stored so a test can call this more than
    once without colliding with the (source, external_task_id) constraint.
    """
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    repo = TaskRepository(session)
    offset = len(repo.list_by_status())
    ids = []
    for index in range(offset, offset + count):
        record = repo.add(
            Task(task_id=f"T-{index}", project_name="Project Dynamo", title=f"Task {index}"),
            project,
        )
        repo.set_status(record, status)
        ids.append(record.id)
    session.commit()
    return ids


# ------------------------------------------------------------------ approval


def test_approve_moves_a_reviewed_task_to_approved(session):
    task_id = seed(session)[0]
    record = ApprovalService(session).approve(task_id, RULES)
    session.commit()
    assert record.status is TaskStatus.APPROVED


def test_approve_rejects_a_task_in_the_wrong_status(session):
    task_id = seed(session, TaskStatus.DISCOVERED)[0]
    with pytest.raises(ApprovalError):
        ApprovalService(session).approve(task_id, RULES)


def test_approve_rejects_an_unknown_task(session):
    with pytest.raises(ApprovalError):
        ApprovalService(session).approve(999, RULES)


def test_approve_blocks_when_the_workload_is_full(session):
    ids = seed(session, count=3)
    service = ApprovalService(session)
    service.approve(ids[0], RULES)
    service.approve(ids[1], RULES)
    session.commit()
    with pytest.raises(ApprovalError):
        service.approve(ids[2], RULES)


def test_force_overrides_the_capacity_block(session):
    ids = seed(session, count=3)
    service = ApprovalService(session)
    service.approve(ids[0], RULES)
    service.approve(ids[1], RULES)
    record = service.approve(ids[2], RULES, force=True)
    session.commit()
    assert record.status is TaskStatus.APPROVED
    assert EventRepository(session).count_by_type("task_approved") == 3


def test_reject_skips_a_task(session):
    task_id = seed(session)[0]
    record = ApprovalService(session).reject(task_id, reason="not interested")
    session.commit()
    assert record.status is TaskStatus.SKIPPED


def test_reject_can_undo_an_approval(session):
    task_id = seed(session)[0]
    service = ApprovalService(session)
    service.approve(task_id, RULES)
    assert service.reject(task_id).status is TaskStatus.SKIPPED


def test_reject_refuses_an_already_skipped_task(session):
    task_id = seed(session, TaskStatus.SKIPPED)[0]
    with pytest.raises(ApprovalError):
        ApprovalService(session).reject(task_id)


def test_reset_requeues_for_evaluation(session):
    task_id = seed(session, TaskStatus.SKIPPED)[0]
    record = ApprovalService(session).reset(task_id)
    session.commit()
    assert record.status is TaskStatus.DISCOVERED


def test_reset_refuses_an_approved_task(session):
    task_id = seed(session)[0]
    service = ApprovalService(session)
    service.approve(task_id, RULES)
    with pytest.raises(ApprovalError):
        service.reset(task_id)


def test_transitions_are_recorded(session):
    ids = seed(session, count=2)
    service = ApprovalService(session)
    service.approve(ids[0], RULES)
    service.reject(ids[1])
    session.commit()

    events = EventRepository(session)
    assert events.count_by_type("task_approved") == 1
    assert events.count_by_type("task_rejected") == 1


# --------------------------------------------------------------------- board


def test_snapshot_counts_by_status(session):
    seed(session, TaskStatus.REVIEW_REQUIRED, 2)
    seed(session, TaskStatus.SKIPPED, 1)
    snapshot = build_snapshot(session, RULES)
    assert snapshot.count(TaskStatus.REVIEW_REQUIRED) == 2
    assert snapshot.count(TaskStatus.SKIPPED) == 1


def test_snapshot_exposes_projects_and_workload(session):
    seed(session)
    snapshot = build_snapshot(session, RULES)
    assert snapshot.projects == ["Project Dynamo"]
    assert snapshot.workload.maximum == 2


def test_snapshot_survives_tasks_with_no_evaluation(session):
    seed(session, TaskStatus.DISCOVERED)
    view = build_snapshot(session, RULES).tasks[0]
    assert view.score is None
    assert view.score_text == "-"


def test_snapshot_carries_evaluation_details(session):
    from core.enums import Decision
    from database.repositories import EvaluationRepository

    task_id = seed(session)[0]
    record = TaskRepository(session).list_by_status()[0]
    EvaluationRepository(session).add(
        record, score=88, decision=Decision.HIGH_MATCH, reason="good fit", risks=["scope"]
    )
    session.commit()

    view = next(t for t in build_snapshot(session, RULES).tasks if t.id == task_id)
    assert view.score == 88
    assert view.decision == "HIGH_MATCH"
    assert view.risks == ["scope"]


def test_snapshot_filters_by_status(session):
    seed(session, TaskStatus.REVIEW_REQUIRED, 2)
    seed(session, TaskStatus.APPROVED, 1)
    snapshot = build_snapshot(session, RULES)
    assert len(snapshot.by_status(TaskStatus.REVIEW_REQUIRED)) == 2
    assert len(snapshot.by_status(TaskStatus.REVIEW_REQUIRED, TaskStatus.APPROVED)) == 3


def test_snapshot_includes_recent_events(session):
    seed(session)
    EventRepository(session).log("agent_started", message="up")
    session.commit()
    assert any(e.event_type == "agent_started" for e in build_snapshot(session, RULES).events)
