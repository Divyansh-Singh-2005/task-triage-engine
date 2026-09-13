"""Run the monitoring loop from the command line.

    python -m scripts.run_agent --once
    python -m scripts.run_agent --interval 300

The loop honours the agent status set in the dashboard: while the agent is
stopped, cycles are skipped rather than the process exiting, so you can leave
this running and toggle the agent from the UI.
"""

from __future__ import annotations

import argparse
import signal
import sys

from agent.orchestrator import AgentOrchestrator
from agent.sources.registry import build_sources
from config.settings import get_settings
from core.logging import configure_logging, get_logger
from services.agent_context import AgentContext

logger = get_logger("run_agent")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Handshake Task Agent monitoring loop")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--interval", type=int, default=None,
                        help="seconds between cycles (default: POLL_INTERVAL_SECONDS)")
    parser.add_argument("--force", action="store_true",
                        help="run even if the agent status is stopped (implies --once)")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(args.log_level or settings.log_level,
                      log_file=settings.log_file or None)

    context = AgentContext(settings)
    sources = build_sources(settings)
    orchestrator = AgentOrchestrator(context, sources)

    logger.info("starting", extra={"event": "startup",
                                   "provider": context.provider.describe(),
                                   "sources": ",".join(s.name for s in sources),
                                   "inbox": str(settings.inbox_path)})

    if args.once or args.force:
        result = orchestrator.run_once(ignore_agent_status=args.force)
        print(result.summary())
        for error in result.errors:
            print(f"  ERROR {error}")
        return 0 if result.ok else 1

    stopping = {"value": False}

    def handle_signal(signum, _frame):
        # Finish the cycle in flight, then exit cleanly.
        logger.info("signal received", extra={"event": "shutdown", "signal": signum})
        stopping["value"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    interval = args.interval or settings.poll_interval_seconds
    orchestrator.run_forever(interval, should_stop=lambda: stopping["value"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
