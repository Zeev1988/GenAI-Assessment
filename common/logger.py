"""Shared logger factory — console + optional rotating file handler."""

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
    """Return a configured logger for `name`, creating it on first call."""
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(min(lvl, logging.DEBUG) if log_file else lvl)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(_FORMAT)
    logger.addHandler(ch)

    if log_file is not None:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(_FORMAT)
            logger.addHandler(fh)
        except OSError as exc:
            logger.warning("Cannot open log file '%s': %s", log_file, exc)

    _registry[name] = logger
    return logger
