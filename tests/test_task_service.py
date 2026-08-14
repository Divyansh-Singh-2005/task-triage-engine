"""Tests for the manual task source and the ingestion pipeline."""

from __future__ import annotations

import json

import pytest

from agent.sources.manual import ManualTaskSource
from core.enums import MonitoringMode, TaskStatus
from core.exceptions import TaskSourceError
from core.schemas import AgentConfig
from database.database import create_db_engine, create_session_factory, init_db
from database.repositories import EventRepository, TaskRepository
from services.task_service import TaskIngestionService


@pytest.fixture
def session(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(engine)
    factory = create_session_factory(engine)
    with factory() as sess:
        yield sess
    engine.dispose()


LISTINGS = [
    {"task_id": "D-1", "project_name": "Project Dynamo", "title": "RDF graph task"},
    {"task_id": "A-1", "project_name": "Project Alpha", "title": "Something else"},
    {"task_id": "D-2", "project_name": "Project Dynamo", "title": "Maze task"},
]


def active_config() -> AgentConfig:
    return AgentConfig(
        monitoring_mode=MonitoringMode.ACTIVE_PROJECT, active_project="Project Dynamo"
    )


# ------------------------------------------------------------ manual source


def test_manual_source_reads_inline_tasks():
    tasks = ManualTaskSource(LISTINGS).fetch()
    assert [t.task_id for t in tasks] == ["D-1", "A-1", "D-2"]
    assert all(t.source == "manual" for t in tasks)


def test_manual_source_reads_a_json_list(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(LISTINGS), encoding="utf-8")
    assert len(ManualTaskSource(path=path).fetch()) == 3


def test_manual_source_reads_a_tasks_key(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": LISTINGS}), encoding="utf-8")
    assert len(ManualTaskSource(path=path).fetch()) == 3


def test_manual_source_synthesizes_a_missing_id():
    tasks = ManualTaskSource([{"project_name": "Project Dynamo", "title": "No id"}]).fetch()
    assert tasks[0].task_id == "manual-No id"


def test_manual_source_raises_on_missing_file(tmp_path):
    with pytest.raises(TaskSourceError):
        ManualTaskSource(path=tmp_path / "nope.json").fetch()


def test_manual_source_raises_on_bad_json(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TaskSourceError):
        ManualTaskSource(path=path).fetch()


def test_manual_source_raises_on_invalid_task():
    with pytest.raises(TaskSourceError):
        ManualTaskSource([{"task_id": "X", "title": "no project"}]).fetch()


# ---------------------------------------------------------------- ingestion


def test_ingest_stores_only_in_scope_tasks(session):
    report = TaskIngestionService(session).ingest(ManualTaskSource(LISTINGS), active_config())
    session.commit()

    assert report.fetched == 3
    assert report.stored == 2
    assert report.filtered_out == 1
    assert report.ok
    assert len(TaskRepository(session).list_by_status(TaskStatus.DISCOVERED)) == 2


def test_any_project_mode_stores_everything(session):
    config = AgentConfig(monitoring_mode=MonitoringMode.ANY_PROJECT)
    report = TaskIngestionService(session).ingest(ManualTaskSource(LISTINGS), config)
    session.commit()
    assert report.stored == 3
    assert report.filtered_out == 0


def test_selected_projects_mode_stores_the_selection(session):
    config = AgentConfig(
        monitoring_mode=MonitoringMode.SELECTED_PROJECTS,
        selected_projects=["Project Alpha"],
    )
    report = TaskIngestionService(session).ingest(ManualTaskSource(LISTINGS), config)
    session.commit()
    assert report.stored == 1
    assert report.filtered_out == 2


def test_second_ingest_of_the_same_batch_stores_nothing(session):
    service = TaskIngestionService(session)
    service.ingest(ManualTaskSource(LISTINGS), active_config())
    session.commit()

    second = service.ingest(ManualTaskSource(LISTINGS), active_config())
    session.commit()

    assert second.stored == 0
    assert second.duplicates == 2  # the filtered one was never stored to dupe
    assert second.filtered_out == 1


def test_duplicate_check_precedes_the_filter(session):
    """A stored task stays stored when the mode narrows underneath it."""
    service = TaskIngestionService(session)
    service.ingest(ManualTaskSource(LISTINGS), AgentConfig(monitoring_mode=MonitoringMode.ANY_PROJECT))
    session.commit()

    narrowed = service.ingest(ManualTaskSource(LISTINGS), active_config())
    session.commit()
    assert narrowed.duplicates == 3
    assert narrowed.filtered_out == 0


def test_one_bad_task_does_not_abort_the_batch(session):
    class HalfBrokenSource(ManualTaskSource):
        def fetch(self):
            tasks = super().fetch()
            tasks[0].project_name = ""  # bypasses validation post-construction
            return tasks

    report = TaskIngestionService(session).ingest(HalfBrokenSource(LISTINGS), active_config())
    session.commit()

    assert len(report.errors) == 1
    assert report.stored == 1  # D-2 still made it through


def test_source_failure_is_reported_not_raised(session):
    class BrokenSource(ManualTaskSource):
        def fetch(self):
            raise TaskSourceError("upstream unavailable")

    report = TaskIngestionService(session).ingest(BrokenSource(), active_config())
    session.commit()

    assert report.ok is False
    assert report.fetched == 0
    assert EventRepository(session).count_by_type("source_failed") == 1


def test_ingestion_writes_an_audit_trail(session):
    TaskIngestionService(session).ingest(ManualTaskSource(LISTINGS), active_config())
    session.commit()

    events = EventRepository(session)
    assert events.count_by_type("task_discovered") == 2
    assert events.count_by_type("task_filtered") == 1
    assert events.count_by_type("ingest_completed") == 1


def test_projects_are_created_from_ingested_tasks(session):
    from database.repositories import ProjectRepository

    TaskIngestionService(session).ingest(
        ManualTaskSource(LISTINGS), AgentConfig(monitoring_mode=MonitoringMode.ANY_PROJECT)
    )
    session.commit()
    assert {p.name for p in ProjectRepository(session).list_all()} == {
        "Project Dynamo",
        "Project Alpha",
    }
