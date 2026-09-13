"""Structured logging with redaction.

Two jobs. First, make log lines machine-greppable: one line per event, with
context as ``key=value`` pairs rather than baked into prose. Second, make it
hard to leak a credential by accident - the redacting filter sits on the
handler, so it catches anything any module logs, including third-party
libraries that were never told about our conventions.

Redaction is a safety net, not a licence. Do not log secrets deliberately.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

#: Attributes ``logging`` puts on every record. Anything else came from
#: ``extra=`` and is treated as structured context.
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
}

REDACTED = "[REDACTED]"

#: Ordered most-specific first. Each pattern keeps any non-secret prefix it
#: matched, so the line stays readable after scrubbing.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), REDACTED),
    (re.compile(r"(?i)\bBearer\s+\S+"), "Bearer " + REDACTED),
    # No leading \b: an underscore is a word character, so "ANTHROPIC_API_KEY"
    # has no boundary before "API". The value runs to end of line rather than
    # one token, so "Authorization: Bearer <jwt>" does not leak the jwt.
    (
        re.compile(
            r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|apikey|token|secret|password|passwd|authorization))"
            r"(\s*[=:]\s*)(.+)"
        ),
        r"\1\2" + REDACTED,
    ),
)


def redact(text: str) -> str:
    """Scrub anything that looks like a credential."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Rewrites messages and structured context before they reach a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _scrub(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(_scrub(a) for a in record.args)
        for key in list(vars(record)):
            if key not in _RESERVED:
                setattr(record, key, _scrub(getattr(record, key)))
        return True


def _scrub(value: Any) -> Any:
    return redact(value) if isinstance(value, str) else value


class StructuredFormatter(logging.Formatter):
    """``time level logger message key=value ...``"""

    default_time_format = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record)} {record.levelname:<7} "
            f"{record.name} {record.getMessage()}"
        )
        context = " ".join(
            f"{key}={_render(getattr(record, key))}"
            for key in sorted(vars(record))
            if key not in _RESERVED
        )
        line = f"{base} {context}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _render(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(
    level: str = "INFO",
    *,
    log_file: str | Path | None = None,
    stream=None,
) -> logging.Logger:
    """Install handlers on the root logger. Safe to call more than once.

    Existing handlers are replaced rather than added to, so a Streamlit rerun
    or a second CLI invocation does not produce duplicated lines.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = StructuredFormatter()
    redactor = SecretRedactingFilter()

    handlers: Iterable[logging.Handler] = [logging.StreamHandler(stream or sys.stderr)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [*handlers, logging.FileHandler(path, encoding="utf-8")]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(redactor)
        root.addHandler(handler)

    # SQLAlchemy and urllib3 are chatty at INFO and say nothing we need.
    for noisy in ("sqlalchemy.engine", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
