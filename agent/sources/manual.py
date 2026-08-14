"""Manual task source: tasks you supply yourself.

This is the source that needs no platform access at all. It reads task
listings from an in-memory list or a JSON file, which means the entire
pipeline downstream of it - filtering, evaluation, scoring, approval,
persistence, dashboard - is fully exercisable today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from core.exceptions import TaskSourceError
from core.schemas import Task
from agent.sources.base import TaskSource


class ManualTaskSource(TaskSource):
    """Yields tasks from a supplied list of dicts and/or a JSON file.

    The JSON file may hold either a bare list of task objects or an object
    with a ``tasks`` key, since both shapes show up when pasting from
    different places.
    """

    name = "manual"

    def __init__(
        self,
        tasks: Iterable[dict[str, Any] | Task] | None = None,
        path: str | Path | None = None,
    ) -> None:
        self._inline = list(tasks or [])
        self.path = Path(path) if path else None

    def fetch(self) -> list[Task]:
        raw: list[dict[str, Any] | Task] = list(self._inline)
        if self.path is not None:
            raw.extend(self._read_file(self.path))
        return [self._coerce(item, index) for index, item in enumerate(raw)]

    def _read_file(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise TaskSourceError(f"task file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaskSourceError(f"{path} is not valid JSON: {exc}") from exc

        if isinstance(data, dict):
            data = data.get("tasks", [])
        if not isinstance(data, list):
            raise TaskSourceError(f"{path} must contain a list of tasks or a 'tasks' key")
        return data

    def _coerce(self, item: dict[str, Any] | Task, index: int) -> Task:
        if isinstance(item, Task):
            return item.model_copy(update={"source": self.name})
        if not isinstance(item, dict):
            raise TaskSourceError(f"task #{index} is {type(item).__name__}, expected an object")

        payload = dict(item)
        payload["source"] = self.name
        # A pasted listing often has no id of its own; fall back to something
        # stable so the duplicate guard still has a key to work with.
        payload.setdefault("task_id", f"manual-{payload.get('title', index)}")
        try:
            return Task.model_validate(payload)
        except Exception as exc:
            raise TaskSourceError(f"task #{index} is invalid: {exc}") from exc

    def add(self, **fields: Any) -> None:
        """Queue one more task, for interactive use from the dashboard."""
        self._inline.append(fields)
