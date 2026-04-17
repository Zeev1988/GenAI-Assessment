"""Structured JSON logging with per-request correlation IDs.

The module is idempotent: calling :func:`configure_logging` multiple times has
no additional effect. A ``correlation_id`` context variable is attached to
every log record so that a single UI request can be traced across all stages.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_configured = False


def _inject_correlation_id(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    event_dict.setdefault("correlation_id", _correlation_id.get())
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib to emit JSON to stderr.

    Safe to call more than once; subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _inject_correlation_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger, configuring logging lazily if needed."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def set_correlation_id(cid: str | None = None) -> str:
    """Set (or generate) a correlation ID for the current context and return it."""
    value = cid or uuid.uuid4().hex
    _correlation_id.set(value)
    return value


def current_correlation_id() -> str:
    return _correlation_id.get()


@contextmanager
def correlation_scope(cid: str | None = None) -> Iterator[str]:
    """Context manager that sets a correlation ID for its body."""
    previous = _correlation_id.get()
    value = set_correlation_id(cid)
    try:
        yield value
    finally:
        _correlation_id.set(previous)
