"""The monitoring loop.

Two guarantees shape everything here:

1. **A cycle never raises.** Whatever fails - a source, the LLM, the database -
   is caught, recorded, and the loop continues. A monitoring agent that dies at
   3am because one HTTP call timed out is worse than useless, because you stop
   checking on it.

2. **Config is re-read every cycle.** Change the mode or stop the agent from
   the dashboard and the next cycle honours it. No restart, no shared state
   between the two processes beyond the config file and the database.

Each source ingests in its own transaction, so one source failing does not
roll back another's work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

from agent.sources.base import TaskSource
from core.enums import AgentStatus
from core.logging import get_logger
from services.agent_context import AgentContext
from services.evaluation_service import EvaluationRun, EvaluationService
from services.task_service import IngestionReport, TaskIngestionService

logger = get_logger(__name__)

#: Backoff applied after consecutive failed cycles, in seconds. The last value
#: repeats indefinitely - the loop slows down but never gives up.
BACKOFF_SECONDS = (30, 60, 300, 900)


@dataclass
class CycleResult:
    started_at: datetime
    ingested: list[IngestionReport] = field(default_factory=list)
    evaluated: EvaluationRun | None = None
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.skipped_reason is None

    @property
    def ok(self) -> bool:
        return not self.errors and all(report.ok for report in self.ingested)

    @property
    def stored(self) -> int:
        return sum(report.stored for report in self.ingested)

    def summary(self) -> str:
        if not self.ran:
            return f"cycle skipped: {self.skipped_reason}"
        parts = [f"stored {self.stored}"]
        if self.evaluated:
            parts.append(self.evaluated.summary())
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        return "; ".join(parts)


class AgentOrchestrator:
    """Runs ingest-then-evaluate cycles against the configured sources."""

    def __init__(self, context: AgentContext, sources: Sequence[TaskSource]) -> None:
        self.context = context
        self.sources = list(sources)

    # ------------------------------------------------------------- one cycle

    def run_once(self, *, ignore_agent_status: bool = False) -> CycleResult:
        """Run a single cycle. Never raises."""
        result = CycleResult(started_at=datetime.now(timezone.utc))

        try:
            config = self.context.config()
        except Exception as exc:
            result.errors.append(f"config unreadable: {exc}")
            logger.error("cycle aborted", extra={"event": "config_error", "detail": str(exc)})
            return result

        if not ignore_agent_status and config.agent_status is not AgentStatus.RUNNING:
            result.skipped_reason = "agent is stopped"
            logger.debug("cycle skipped", extra={"event": "agent_stopped"})
            return result

        if not self.sources:
            result.skipped_reason = "no sources configured"
            logger.warning("cycle skipped", extra={"event": "no_sources"})
            return result

        for source in self.sources:
            try:
                with self.context.session() as session:
                    report = TaskIngestionService(session).ingest(source, config)
                result.ingested.append(report)
                logger.info("ingested", extra={"event": "ingest", "source": source.name,
                                               "stored": report.stored,
                                               "duplicates": report.duplicates,
                                               "filtered_out": report.filtered_out})
            except Exception as exc:
                result.errors.append(f"{source.name}: {exc}")
                logger.exception("ingest failed", extra={"event": "ingest_failed",
                                                        "source": source.name})

        try:
            with self.context.session() as session:
                result.evaluated = EvaluationService(session, self.context.provider).run(
                    config, self.context.profile
                )
            logger.info("evaluated", extra={"event": "evaluate",
                                            "count": result.evaluated.evaluated,
                                            "review": result.evaluated.review_required})
        except Exception as exc:
            result.errors.append(f"evaluation: {exc}")
            logger.exception("evaluation failed", extra={"event": "evaluate_failed"})

        self._record_cycle(result)
        return result

    def _record_cycle(self, result: CycleResult) -> None:
        """Write the cycle to the audit trail.

        Best-effort: if the database is the thing that is broken, the loop
        still has to survive, so a failure here is logged and swallowed.
        """
        try:
            with self.context.session() as session:
                from database.repositories import EventRepository

                EventRepository(session).log(
                    "cycle_completed",
                    message=result.summary(),
                    payload={
                        "stored": result.stored,
                        "sources": [r.source for r in result.ingested],
                        "errors": len(result.errors),
                        "ok": result.ok,
                    },
                )
        except Exception as exc:
            logger.warning("could not record cycle", extra={"event": "cycle_record_failed",
                                                            "detail": str(exc)})

    # ------------------------------------------------------------- main loop

    def run_forever(
        self,
        interval: int,
        *,
        max_cycles: int | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[CycleResult]:
        """Loop until stopped.

        ``sleeper``, ``max_cycles`` and ``should_stop`` exist so the loop is
        testable without real time passing or a real signal handler.
        """
        results: list[CycleResult] = []
        consecutive_failures = 0
        cycle = 0

        logger.info("agent loop starting", extra={"event": "loop_start", "interval": interval,
                                                  "sources": len(self.sources)})

        while max_cycles is None or cycle < max_cycles:
            if should_stop is not None and should_stop():
                logger.info("agent loop stopping", extra={"event": "loop_stop"})
                break

            cycle += 1
            result = self.run_once()
            results.append(result)

            if result.ran and not result.ok:
                consecutive_failures += 1
            elif result.ran:
                consecutive_failures = 0

            delay = self._delay(interval, consecutive_failures)
            if consecutive_failures:
                logger.warning("backing off", extra={"event": "backoff",
                                                     "failures": consecutive_failures,
                                                     "delay": delay})
            if max_cycles is None or cycle < max_cycles:
                sleeper(delay)

        return results

    @staticmethod
    def _delay(interval: int, consecutive_failures: int) -> float:
        """Normal interval when healthy; escalating backoff when not."""
        if consecutive_failures <= 0:
            return float(interval)
        index = min(consecutive_failures, len(BACKOFF_SECONDS)) - 1
        return float(max(interval, BACKOFF_SECONDS[index]))
