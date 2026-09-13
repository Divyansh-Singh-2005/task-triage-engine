"""Application wiring.

One object owns the expensive, long-lived pieces - engine, session factory,
config manager, LLM provider - so the dashboard, the CLI scripts and any
future scheduler all build the application the same way instead of each
assembling their own half-correct version.

Deliberately free of Streamlit imports: this must stay usable from a plain
script and a test.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from agent.project_manager import ProjectManager
from config.settings import Settings, get_settings
from core.profile import SkillProfile, load_profile
from core.schemas import AgentConfig
from database.database import create_db_engine, create_session_factory, init_db, session_scope
from integrations.llm.base import LLMProvider
from integrations.llm.factory import get_provider


class AgentContext:
    """Everything the application needs, built once."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.manager = ProjectManager(self.settings.config_path, self.settings.defaults_path)
        self.engine = create_db_engine(self.settings.resolved_database_url())
        init_db(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self._provider: LLMProvider | None = None
        self._profile: SkillProfile | None = None

    # ------------------------------------------------------------- accessors

    def config(self, *, force: bool = True) -> AgentConfig:
        """Current config.

        Re-read from disk by default: the dashboard reruns constantly and a
        stale in-memory copy would silently show the wrong mode after an edit.
        """
        return self.manager.load(force=force)

    @property
    def profile(self) -> SkillProfile:
        if self._profile is None:
            self._profile = load_profile(self.settings.profile_path)
        return self._profile

    def reload_profile(self) -> SkillProfile:
        self._profile = load_profile(self.settings.profile_path)
        return self._profile

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider(self.settings)
        return self._provider

    @property
    def provider_is_fallback(self) -> bool:
        """True when a model provider was requested but no key was found."""
        requested = (self.settings.llm_provider or "heuristic").strip().casefold()
        return requested != "heuristic" and self.provider.name == "heuristic"

    # -------------------------------------------------------------- sessions

    @contextmanager
    def session(self) -> Iterator[Session]:
        with session_scope(self.session_factory) as session:
            yield session
