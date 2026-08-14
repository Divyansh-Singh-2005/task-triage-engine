"""Manual end-to-end run of the ingestion pipeline.

Usage:
    python -m scripts.demo_ingest
    python -m scripts.demo_ingest tasks.json

With no argument it uses a built-in sample batch. Run it twice to watch the
duplicate guard work.
"""

from __future__ import annotations

import sys

from agent.project_manager import ProjectManager
from agent.sources.manual import ManualTaskSource
from config.settings import get_settings
from database.database import create_db_engine, create_session_factory, init_db, session_scope
from database.repositories import EventRepository, TaskRepository
from services.task_service import TaskIngestionService

SAMPLE = [
    {
        "task_id": "D-1",
        "project_name": "Project Dynamo",
        "title": "RDF beneficial-ownership task",
        "description": "Resolve layered semantic rules over a synthetic registry graph.",
        "requirements": ["python", "rdflib", "exact arithmetic"],
        "reward": "$120",
    },
    {
        "task_id": "A-1",
        "project_name": "Project Alpha",
        "title": "Unrelated work item",
    },
    {
        "task_id": "D-2",
        "project_name": "project  dynamo",
        "title": "Modular-transition labyrinth task",
        "requirements": ["python", "graph search"],
    },
]


def main() -> int:
    settings = get_settings()
    manager = ProjectManager(settings.config_path, settings.defaults_path)
    config = manager.load()

    engine = create_db_engine(settings.resolved_database_url())
    init_db(engine)
    factory = create_session_factory(engine)

    source = (
        ManualTaskSource(path=sys.argv[1]) if len(sys.argv) > 1 else ManualTaskSource(SAMPLE)
    )

    print(f"mode={config.monitoring_mode.value}  active={config.active_project!r}")
    print(f"source={source.describe()}\n")

    with session_scope(factory) as session:
        report = TaskIngestionService(session).ingest(source, config)
        print(report.summary())
        for task_id, reason in report.filter_reasons.items():
            print(f"  {task_id}: {reason}")
        for error in report.errors:
            print(f"  ERROR {error}")

    with session_scope(factory) as session:
        print("\nstored tasks:")
        for record in TaskRepository(session).list_by_status():
            print(f"  [{record.status.value}] {record.external_task_id} - {record.title}")
            print(f"      project={record.project.name}")

        print("\nrecent events:")
        for event in EventRepository(session).recent(8):
            print(f"  {event.event_type}: {event.message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
