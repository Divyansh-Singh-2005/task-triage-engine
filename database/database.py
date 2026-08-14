"""Database engine and session management.

SQLite is the Phase 1 target, but nothing here is SQLite-specific beyond the
connect args, so switching ``DATABASE_URL`` to a PostgreSQL DSN is the only
change needed to migrate.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import DateTime, Engine, TypeDecorator, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class UtcDateTime(TypeDecorator):
    """A DateTime that is always timezone-aware UTC in Python.

    SQLite silently discards tzinfo, so a naive datetime read back from the
    database would compare unequal to the aware one that was written. This
    normalizes on the way in and re-attaches UTC on the way out, which keeps
    deadline comparisons honest regardless of backend.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_db_engine(url: str, *, echo: bool = False) -> Engine:
    """Build an engine with SQLite-appropriate defaults.

    ``check_same_thread=False`` is needed because Streamlit runs callbacks on
    threads other than the one that created the connection.
    """
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            # Foreign keys are OFF by default in SQLite; without this the
            # task -> project relationship is not actually enforced.
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    """Create any missing tables.

    Adequate until the schema starts changing under real data; at that point
    this gets replaced with Alembic migrations.
    """
    from database import models  # noqa: F401  (registers models on Base)

    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
