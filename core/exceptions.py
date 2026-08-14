"""Exception hierarchy.

Everything the application raises deliberately descends from ``AgentError``,
so the monitoring loop can catch that one class and keep running rather than
catching bare ``Exception`` and swallowing real bugs.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all deliberate errors raised by this application."""


class ConfigError(AgentError):
    """The configuration is invalid or could not be loaded/saved."""


class UnknownProjectError(ConfigError):
    """A project was referenced that the agent has never seen."""

    def __init__(self, name: str, known: list[str] | None = None) -> None:
        self.name = name
        self.known = known or []
        known_text = ", ".join(self.known) if self.known else "none registered"
        super().__init__(f"unknown project {name!r} (known: {known_text})")


class TaskSourceError(AgentError):
    """A task source failed to produce tasks."""


class EvaluationError(AgentError):
    """The LLM evaluation failed or returned unusable output."""


class ClaimError(AgentError):
    """A claim attempt failed."""
