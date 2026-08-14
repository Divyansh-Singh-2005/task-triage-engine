"""Run the full pipeline: ingest, then evaluate and decide.

    python -m scripts.demo_evaluate

Safe to run repeatedly: ingestion de-duplicates and evaluation only picks up
tasks still in DISCOVERED.
"""

from __future__ import annotations

from agent.project_manager import ProjectManager
from agent.sources.manual import ManualTaskSource
from config.settings import get_settings
from core.profile import load_profile
from database.database import create_db_engine, create_session_factory, init_db, session_scope
from database.repositories import TaskRepository
from integrations.llm.factory import get_provider
from scripts.demo_ingest import SAMPLE
from services.evaluation_service import EvaluationService
from services.task_service import TaskIngestionService


def main() -> int:
    settings = get_settings()
    config = ProjectManager(settings.config_path, settings.defaults_path).load()
    profile = load_profile(settings.profile_path)
    provider = get_provider(settings)

    engine = create_db_engine(settings.resolved_database_url())
    init_db(engine)
    factory = create_session_factory(engine)

    print(f"provider={provider.describe()}  profile={profile.display_name}")
    print(f"mode={config.monitoring_mode.value}  active={config.active_project!r}")
    print(f"rules: review>={config.task_rules.review_threshold} "
          f"high>={config.task_rules.minimum_ai_score} "
          f"max_active={config.task_rules.maximum_active_tasks} "
          f"approval={config.task_rules.require_human_approval}\n")

    with session_scope(factory) as session:
        print(TaskIngestionService(session).ingest(ManualTaskSource(SAMPLE), config).summary())

    with session_scope(factory) as session:
        run = EvaluationService(session, provider).run(config, profile)
        print(run.summary())
        for error in run.errors:
            print(f"  ERROR {error}")

    with session_scope(factory) as session:
        print("\ntask board:")
        for record in TaskRepository(session).list_by_status():
            latest = record.latest_evaluation
            score = f"score={latest.score:>3}" if latest else "score=  -"
            print(f"  [{record.status.value:<16}] {score}  {record.title}")
            if latest and latest.risks:
                for risk in latest.risks:
                    print(f"       risk: {risk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
