"""Tests for structured logging, the inbox source, and the monitoring loop."""

from __future__ import annotations

import io
import json
import logging

import pytest

from agent.orchestrator import BACKOFF_SECONDS, AgentOrchestrator
from agent.sources.base import TaskSource
from agent.sources.inbox import InboxSource
from agent.sources.registry import build_sources
from core.enums import AgentStatus, MonitoringMode, TaskStatus
from core.exceptions import TaskSourceError
from core.logging import StructuredFormatter, configure_logging, get_logger, redact
from config.settings import Settings
from database.repositories import EventRepository, TaskRepository
from services.agent_context import AgentContext

LISTINGS = [
    {"task_id": "D-1", "project_name": "Project Dynamo", "title": "RDF graph task"},
    {"task_id": "A-1", "project_name": "Project Alpha", "title": "Something else"},
]


# ------------------------------------------------------------------- logging


@pytest.mark.parametrize(
    "raw",
    [
        "using sk-ant-api03-abcdefghijklmnopqrstuvwxyz123456",
        "token=ghp_abcdefghijklmnopqrstuvwxyz1234",
        "ANTHROPIC_API_KEY=supersecretvalue",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload",
        'password: "hunter2"',
    ],
)
def test_credentials_are_redacted(raw):
    scrubbed = redact(raw)
    assert "[REDACTED]" in scrubbed
    for secret in ("supersecretvalue", "hunter2", "ghp_abcdefghijklmnopqrstuvwxyz1234"):
        assert secret not in scrubbed


def test_redaction_keeps_the_readable_prefix():
    assert redact("api_key=abc123def456").startswith("api_key=")


def test_ordinary_text_is_untouched():
    message = "evaluated 3 tasks in 1.2s"
    assert redact(message) == message


def test_handler_redacts_whatever_any_module_logs():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("some.third.party").info("connecting with api_key=leakedvalue123")
    assert "leakedvalue123" not in stream.getvalue()
    assert "[REDACTED]" in stream.getvalue()


def test_extra_context_is_redacted_too():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("t").info("call", extra={"header": "Bearer abcdefghijklmnop"})
    assert "abcdefghijklmnop" not in stream.getvalue()


def test_structured_context_is_rendered_as_key_values():
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("t").info("ingested", extra={"event": "ingest", "stored": 3})
    output = stream.getvalue()
    assert "event=ingest" in output
    assert "stored=3" in output


def test_values_with_spaces_are_quoted():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    record.detail = "two words"
    assert 'detail="two words"' in StructuredFormatter().format(record)


def test_configure_logging_does_not_stack_handlers():
    configure_logging("INFO", stream=io.StringIO())
    configure_logging("INFO", stream=io.StringIO())
    assert len(logging.getLogger().handlers) == 1


# -------------------------------------------------------------- inbox source


def test_inbox_returns_nothing_when_the_file_is_absent(tmp_path):
    assert InboxSource(tmp_path / "inbox.json").fetch() == []


def test_inbox_reads_tasks_when_present(tmp_path):
    path = tmp_path / "inbox.json"
    path.write_text(json.dumps(LISTINGS), encoding="utf-8")
    tasks = InboxSource(path).fetch()
    assert [t.task_id for t in tasks] == ["D-1", "A-1"]
    assert all(t.source == "inbox" for t in tasks)


def test_inbox_still_raises_on_a_malformed_file(tmp_path):
    path = tmp_path / "inbox.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(TaskSourceError):
        InboxSource(path).fetch()


def test_registry_builds_the_inbox_source(tmp_path):
    settings = Settings(data_dir=tmp_path)
    sources = build_sources(settings)
    assert [s.name for s in sources] == ["inbox"]


# -------------------------------------------------------------- orchestrator


@pytest.fixture
def context(tmp_path):
    defaults = tmp_path / "defaults.json"
    defaults.write_text(
        json.dumps(
            {
                "agent_status": "running",
                "monitoring_mode": "ACTIVE_PROJECT",
                "active_project": "Project Dynamo",
                "known_projects": ["Project Dynamo"],
                "task_rules": {"minimum_ai_score": 80, "review_threshold": 60,
                               "maximum_active_tasks": 2, "require_human_approval": True},
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path, defaults_path=defaults,
        profile_path=tmp_path / "missing-profile.json", llm_provider="heuristic",
    )
    settings.ensure_dirs()
    return AgentContext(settings)


class ListSource(TaskSource):
    name = "listy"

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def fetch(self):
        from core.schemas import Task

        self.calls += 1
        return [Task(**{**row, "source": self.name}) for row in self.rows]


class BrokenSource(TaskSource):
    name = "broken"

    def fetch(self):
        raise RuntimeError("upstream on fire")


def test_cycle_ingests_and_evaluates(context):
    result = AgentOrchestrator(context, [ListSource(LISTINGS)]).run_once()
    assert result.ran
    assert result.stored == 1  # Project Alpha is filtered out
    assert result.evaluated.evaluated == 1
    assert result.ok


def test_cycle_is_skipped_while_the_agent_is_stopped(context):
    context.manager.set_agent_status(AgentStatus.STOPPED)
    source = ListSource(LISTINGS)
    result = AgentOrchestrator(context, [source]).run_once()
    assert not result.ran
    assert result.skipped_reason == "agent is stopped"
    assert source.calls == 0


def test_force_overrides_the_stopped_status(context):
    context.manager.set_agent_status(AgentStatus.STOPPED)
    result = AgentOrchestrator(context, [ListSource(LISTINGS)]).run_once(ignore_agent_status=True)
    assert result.ran
    assert result.stored == 1


def test_cycle_with_no_sources_is_skipped(context):
    result = AgentOrchestrator(context, []).run_once()
    assert result.skipped_reason == "no sources configured"


def test_a_source_that_explodes_does_not_kill_the_cycle(context):
    good = ListSource(LISTINGS)
    result = AgentOrchestrator(context, [BrokenSource(), good]).run_once()
    assert result.ran
    assert result.ok is False
    assert result.stored == 1  # the healthy source still ran
    assert any("broken" in e for e in result.errors)


def test_cycle_never_raises_even_with_an_unreadable_config(context, tmp_path):
    (tmp_path / "agent_config.json").write_text("{ nope", encoding="utf-8")
    context.manager._config = None
    result = AgentOrchestrator(context, [ListSource(LISTINGS)]).run_once()
    assert not result.ran or result.errors
    assert result.errors


def test_cycle_is_recorded_in_the_audit_trail(context):
    AgentOrchestrator(context, [ListSource(LISTINGS)]).run_once()
    with context.session() as session:
        assert EventRepository(session).count_by_type("cycle_completed") == 1


def test_config_is_reread_between_cycles(context):
    orchestrator = AgentOrchestrator(context, [ListSource(LISTINGS)])
    orchestrator.run_once()
    context.manager.set_monitoring_mode(MonitoringMode.ANY_PROJECT)
    orchestrator.run_once()
    with context.session() as session:
        stored = {t.external_task_id for t in TaskRepository(session).list_by_status()}
    assert stored == {"D-1", "A-1"}  # Alpha only got in after the mode widened


def test_second_cycle_stores_nothing_new(context):
    orchestrator = AgentOrchestrator(context, [ListSource(LISTINGS)])
    orchestrator.run_once()
    second = orchestrator.run_once()
    assert second.stored == 0
    assert second.ingested[0].duplicates == 1


def test_evaluated_tasks_land_in_review(context):
    AgentOrchestrator(context, [ListSource(LISTINGS)]).run_once()
    with context.session() as session:
        statuses = {t.status for t in TaskRepository(session).list_by_status()}
    assert statuses <= {TaskStatus.REVIEW_REQUIRED, TaskStatus.SKIPPED}


# ----------------------------------------------------------------- the loop


def test_loop_runs_the_requested_number_of_cycles(context):
    slept: list[float] = []
    results = AgentOrchestrator(context, [ListSource(LISTINGS)]).run_forever(
        interval=1, max_cycles=3, sleeper=slept.append
    )
    assert len(results) == 3
    assert slept == [1.0, 1.0]  # no sleep after the final cycle


def test_loop_stops_when_asked(context):
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2

    results = AgentOrchestrator(context, [ListSource(LISTINGS)]).run_forever(
        interval=1, max_cycles=10, sleeper=lambda _: None, should_stop=should_stop
    )
    assert len(results) == 2


def test_loop_backs_off_after_repeated_failures(context):
    slept: list[float] = []
    AgentOrchestrator(context, [BrokenSource()]).run_forever(
        interval=1, max_cycles=3, sleeper=slept.append
    )
    assert slept == [float(BACKOFF_SECONDS[0]), float(BACKOFF_SECONDS[1])]


def test_backoff_never_drops_below_the_interval(context):
    orchestrator = AgentOrchestrator(context, [])
    assert orchestrator._delay(600, 1) == 600.0
    assert orchestrator._delay(1, 0) == 1.0


def test_loop_recovers_its_cadence_after_a_success(context):
    slept: list[float] = []
    sources = [BrokenSource()]
    orchestrator = AgentOrchestrator(context, sources)

    def sleeper(delay):
        slept.append(delay)
        sources[:] = [ListSource(LISTINGS)]
        orchestrator.sources = sources

    orchestrator.run_forever(interval=5, max_cycles=3, sleeper=sleeper)
    assert slept[0] == float(BACKOFF_SECONDS[0])
    assert slept[1] == 5.0  # back to normal once a cycle succeeded
