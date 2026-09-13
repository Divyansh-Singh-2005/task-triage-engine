"""A watched JSON file as a task source.

The difference from :class:`ManualTaskSource`: a missing inbox file is a quiet
day, not an outage. The scheduler polls this every cycle, and raising on "no
file yet" would turn an empty queue into a stream of error events.

A file that exists but is malformed is still an error, because that is a real
problem you want to hear about.
"""

from __future__ import annotations

from pathlib import Path

from agent.sources.manual import ManualTaskSource
from core.schemas import Task


class InboxSource(ManualTaskSource):
    """Reads tasks from a JSON file that may not exist yet."""

    name = "inbox"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path=path)

    def fetch(self) -> list[Task]:
        if self.path is None or not self.path.exists():
            return []
        return super().fetch()
