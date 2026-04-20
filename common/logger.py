"""Shared logger factory for both Part 1 and Part 2.

Produces per-name loggers with a console handler at the configured level
plus an optional rotating-file handler (captures everything at DEBUG, so
the file is a durable trace even when the console is set to INFO).

Usage:

    from common import get_logger
    log = get_logger(__name__)
    log.info("something happened")

Settings resolution order:
    1. Explicit arguments to `get_logger`, if provided.
    2. Otherwise falls back to sensible module-level defaults
       (INFO to stdout, no file handler) so the module has no runtime
       dependency on either sub-project's settings.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_registry: dict[str, logging.Logger] = {}
_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(
    name: str,
    *,
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Return a configured logger for *name*, creating it on first call.

    Args:
        name:      Logger name (typically `__name__`).
        level:     Console log level (file handler always captures DEBUG).
        log_file:  Path for a rotating-file handler (5 MB × 3 files).  When
                   omitted, no file handler is attached.
    """
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(min(lvl, logging.DEBUG) if log_file else lvl)
    logger.propagate = False  # don't double-log to root

    # ── Console handler ────────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(_FORMAT)
    logger.addHandler(ch)

    # ── Rotating file handler (optional) ──────────────────────────────────────
    if log_file is not None:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                path,
                maxBytes=5 * 1024 * 1024,  # 5 MB per file
                backupCount=3,
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(_FORMAT)
            logger.addHandler(fh)
        except OSError as exc:
            logger.warning("Cannot open log file '%s': %s", log_file, exc)

    _registry[name] = logger
    return logger
