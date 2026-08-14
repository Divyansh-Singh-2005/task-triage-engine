"""Tests for the ORM models and repositories."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from core.enums import Decision, TaskStatus
from core.schemas import Task
from database.database import create_db_engine, create_session_factory, init_db
from database.repositories import (
    EvaluationRepository,
    EventRepository,
    ProjectRepository,
    TaskRepository,
)


@pytest.fixture
def session(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as sess:
        yield sess
    engine.dispose()


def make_task(project="Project Dynamo", task_id="T-1", **kw) -> Task:
    return Task(task_id=task_id, project_name=project, title=f"Task {task_id}", **kw)


# ------------------------------------------------------------------ projects


def test_get_or_create_is_idempotent(session):
    repo = ProjectRepository(session)
    first = repo.get_or_create("Project Dynamo")
    second = repo.get_or_create("Project Dynamo")
    assert first.id == second.id
    assert len(repo.list_all()) == 1


def test_get_or_create_matches_case_insensitively(session):
    repo = ProjectRepository(session)
    first = repo.get_or_create("Project Dynamo")
    second = repo.get_or_create("  project   DYNAMO ")
    assert first.id == second.id
    assert first.name == "Project Dynamo"  # first display form wins


def test_set_active_clears_previous_active(session):
    repo = ProjectRepository(session)
    repo.get_or_create("Project Alpha")
    repo.set_active("Project Alpha")
    repo.set_active("Project Dynamo")
    active = [p.name for p in repo.list_all() if p.is_active]
    assert active == ["Project Dynamo"]


# --------------------------------------------------------------------- tasks


def test_add_and_read_back_a_task(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    deadline = datetime.now(timezone.utc) + timedelta(days=3)
    record = TaskRepository(session).add(
        make_task(deadline=deadline, requirements=["python", "rdflib"]), project
    )
    session.commit()

    assert record.status is TaskStatus.DISCOVERED
    assert record.requirements == ["python", "rdflib"]
    # UtcDateTime must give back an aware datetime, not a naive one.
    assert record.deadline.tzinfo is not None
    assert record.deadline == deadline


def test_duplicate_external_id_is_rejected_by_the_database(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    repo = TaskRepository(session)
    repo.add(make_task(task_id="T-1"), project)
    session.commit()
    with pytest.raises(IntegrityError):
        repo.add(make_task(task_id="T-1"), project)
        session.commit()


def test_find_duplicate_matches_on_external_id(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    repo = TaskRepository(session)
    stored = repo.add(make_task(task_id="T-1"), project)
    session.commit()
    assert repo.find_duplicate(make_task(task_id="T-1")).id == stored.id


def test_find_duplicate_returns_none_for_new_task(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    TaskRepository(session).add(make_task(task_id="T-1"), project)
    session.commit()
    assert TaskRepository(session).find_duplicate(make_task(task_id="T-2")) is None


def test_count_active_only_counts_committed_slots(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    repo = TaskRepository(session)
    discovered = repo.add(make_task(task_id="T-1"), project)
    approved = repo.add(make_task(task_id="T-2"), project)
    repo.set_status(approved, TaskStatus.APPROVED)
    session.commit()

    assert repo.count_active() == 1
    repo.set_status(discovered, TaskStatus.CLAIMED)
    assert repo.count_active() == 2


def test_list_by_status_filters(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    repo = TaskRepository(session)
    repo.add(make_task(task_id="T-1"), project)
    skipped = repo.add(make_task(task_id="T-2"), project)
    repo.set_status(skipped, TaskStatus.SKIPPED)
    session.commit()

    assert len(repo.list_by_status(TaskStatus.DISCOVERED)) == 1
    assert len(repo.list_by_status()) == 2


# --------------------------------------------------------------- evaluations


def test_evaluations_are_append_only(session):
    project = ProjectRepository(session).get_or_create("Project Dynamo")
    task = TaskRepository(session).add(make_task(), project)
    repo = EvaluationRepository(session)
    repo.add(task, score=70, decision=Decision.SKIP, reason="first pass")
    repo.add(task, score=91, decision=Decision.HIGH_MATCH, reason="second pass")
    session.commit()

    assert len(task.evaluations) == 2
    assert repo.latest_for(task.id).score == 91


# --------------------------------------------------------------------- events


def test_events_are_recorded_and_readable(session):
    repo = EventRepository(session)
    repo.log("agent_started", message="up")
    repo.log("task_filtered", message="wrong project", project_name="Project Alpha")
    session.commit()

    recent = repo.recent()
    assert {e.event_type for e in recent} == {"agent_started", "task_filtered"}
    assert repo.count_by_type("agent_started") == 1
