"""Environment-derived settings.

Paths and secrets live here; behavioural configuration (modes, projects,
rules) lives in the JSON config owned by :class:`ProjectManager`. Keeping the
two separate means the dashboard can rewrite behaviour at runtime without ever
touching credentials.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    config_filename: str = "agent_config.json"
    defaults_path: Path = Field(default=PROJECT_ROOT / "config" / "defaults.json")
    profile_path: Path = Field(default=PROJECT_ROOT / "config" / "profile.json")
    inbox_filename: str = "inbox.json"

    database_url: str = Field(default="")
    log_level: str = "INFO"
    log_file: str = ""
    poll_interval_seconds: int = 300

    # LLM layer (Phase 2). Never hard-code a key; read it from the environment.
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-5"
    llm_max_tokens: int = 1500
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @property
    def config_path(self) -> Path:
        return self.data_dir / self.config_filename

    @property
    def inbox_path(self) -> Path:
        return self.data_dir / self.inbox_filename

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'agent.db').as_posix()}"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
