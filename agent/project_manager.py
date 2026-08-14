"""Project manager: the single owner of agent configuration.

Nothing else in the application writes the config file. The dashboard, the
orchestrator and the CLI all go through this class, which means every change
is validated and every write is atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from core.enums import AgentStatus, MonitoringMode
from core.exceptions import ConfigError, UnknownProjectError
from core.naming import normalize_project_name, project_key
from core.schemas import AgentConfig, TaskRules


class ProjectManager:
    """Loads, validates, mutates and persists an :class:`AgentConfig`.

    The config is held in memory and written back on every successful change,
    so a crash never leaves the dashboard and the file disagreeing.
    """

    def __init__(self, config_path: str | Path, defaults_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path)
        self.defaults_path = Path(defaults_path) if defaults_path else None
        self._config: AgentConfig | None = None

    # ---------------------------------------------------------------- loading

    def load(self, *, force: bool = False) -> AgentConfig:
        """Return the config, reading from disk on first call.

        Falls back to ``defaults_path`` if the config file does not exist yet,
        and to library defaults if that is missing too.
        """
        if self._config is not None and not force:
            return self._config

        raw = self._read_json(self.config_path)
        if raw is None and self.defaults_path is not None:
            raw = self._read_json(self.defaults_path)
        if raw is None:
            raw = {}

        try:
            self._config = AgentConfig.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(f"invalid configuration in {self.config_path}: {exc}") from exc
        return self._config

    @staticmethod
    def _read_json(path: Path | None) -> dict | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc

    @property
    def config(self) -> AgentConfig:
        return self.load()

    def get_active_configuration(self) -> AgentConfig:
        """Public read accessor, per the spec's conceptual API."""
        return self.load()

    # ---------------------------------------------------------------- saving

    def save(self) -> None:
        """Write the config atomically.

        A temp file in the same directory plus ``os.replace`` means a reader
        never observes a half-written file, on POSIX or Windows.
        """
        config = self.load()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.model_dump(mode="json"), indent=2) + "\n"

        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.config_path.parent), prefix=".config-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.config_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _apply(self, **changes: object) -> AgentConfig:
        """Validate a candidate config before committing it.

        Building a fresh model means a rejected change leaves the in-memory
        config untouched, rather than half-applied.
        """
        current = self.load().model_dump()
        current.update(changes)
        try:
            candidate = AgentConfig.model_validate(current)
        except Exception as exc:
            raise ConfigError(f"rejected configuration change: {exc}") from exc
        self._config = candidate
        self.save()
        return candidate

    # ------------------------------------------------------------- projects

    def register_project(self, name: str) -> AgentConfig:
        """Add a project to the known list without selecting it."""
        display = normalize_project_name(name)
        known = list(self.load().known_projects)
        if project_key(display) not in {project_key(k) for k in known}:
            known.append(display)
        return self._apply(known_projects=known)

    def _require_known(self, name: str) -> str:
        display = normalize_project_name(name)
        known = self.load().known_projects
        if project_key(display) not in {project_key(k) for k in known}:
            raise UnknownProjectError(display, known)
        return display

    def set_active_project(self, name: str, *, auto_register: bool = True) -> AgentConfig:
        display = normalize_project_name(name)
        if auto_register:
            self.register_project(display)
        else:
            display = self._require_known(display)
        return self._apply(active_project=display)

    def add_selected_project(self, name: str, *, auto_register: bool = True) -> AgentConfig:
        display = normalize_project_name(name)
        if not auto_register:
            display = self._require_known(display)
        selected = list(self.load().selected_projects) + [display]
        return self._apply(selected_projects=selected)

    def remove_selected_project(self, name: str) -> AgentConfig:
        key = project_key(name)
        selected = [p for p in self.load().selected_projects if project_key(p) != key]
        return self._apply(selected_projects=selected)

    # ----------------------------------------------------------------- modes

    def set_monitoring_mode(self, mode: MonitoringMode | str) -> AgentConfig:
        try:
            resolved = MonitoringMode(mode)
        except ValueError as exc:
            valid = ", ".join(m.value for m in MonitoringMode)
            raise ConfigError(f"unknown monitoring mode {mode!r} (valid: {valid})") from exc
        return self._apply(monitoring_mode=resolved)

    def set_agent_status(self, status: AgentStatus | str) -> AgentConfig:
        try:
            resolved = AgentStatus(status)
        except ValueError as exc:
            raise ConfigError(f"unknown agent status {status!r}") from exc
        return self._apply(agent_status=resolved)

    def set_task_rules(self, **rules: object) -> AgentConfig:
        merged = self.load().task_rules.model_dump()
        merged.update(rules)
        try:
            validated = TaskRules.model_validate(merged)
        except Exception as exc:
            raise ConfigError(f"rejected task rules: {exc}") from exc
        return self._apply(task_rules=validated.model_dump())
