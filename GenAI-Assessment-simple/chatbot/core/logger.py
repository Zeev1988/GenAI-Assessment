"""Centralised logging for the chatbot service.

Creates per-name loggers with both a console handler and a rotating file
handler.  All configuration is driven by ChatBotSettings so the caller
never needs to touch the logging module directly.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_registry: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*, creating it on first call."""
    if name in _registry:
        return _registry[name]

    # Lazy import avoids circular dependency at module-load time.
    from .config import get_settings

    settings = get_settings()

    logger = logging.getLogger(name)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # don't double-log to the root logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ──────────────────────────────────────────────────
    try:
        log_path = Path(settings.log_file)
        # Ensure the parent directory exists.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)  # always capture everything in the file
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Cannot open log file '%s': %s", settings.log_file, exc)

    _registry[name] = logger
    return logger
