"""Persistent disk cache shared by features.

Uses :mod:`diskcache` for a crash-safe, process-safe key/value store that
survives restarts. Keys are prefixed per stage so the same file hash can
cache independent results for OCR and extraction without collision.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from diskcache import Cache

from common.config import Settings, get_settings
from common.logging_config import get_logger

_log = get_logger(__name__)

_STAGE_TTL_SECONDS: dict[str, int] = {
    "ocr": 30 * 24 * 3600,
    "extract": 30 * 24 * 3600,
    "report": 30 * 24 * 3600,
    "embedding": 30 * 24 * 3600,
}


class ResultCache:
    """Thin typed wrapper over :class:`diskcache.Cache`."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        path = cache_dir or settings.cache_path
        path.mkdir(parents=True, exist_ok=True)
        self._cache: Cache = Cache(directory=str(path))

    @staticmethod
    def fingerprint(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _key(stage: str, fingerprint: str, suffix: str = "") -> str:
        suffix_part = f":{suffix}" if suffix else ""
        return f"{stage}:{fingerprint}{suffix_part}"

    def get(self, stage: str, fingerprint: str, suffix: str = "") -> Any | None:
        value = self._cache.get(self._key(stage, fingerprint, suffix), default=None)
        if value is not None:
            _log.debug("cache.hit", stage=stage, fingerprint=fingerprint[:12])
        return value

    def set(self, stage: str, fingerprint: str, value: Any, suffix: str = "") -> None:
        ttl = _STAGE_TTL_SECONDS.get(stage)
        self._cache.set(
            self._key(stage, fingerprint, suffix),
            value,
            expire=ttl,
        )

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> ResultCache:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


_default_cache: ResultCache | None = None


def get_cache(settings: Settings | None = None) -> ResultCache:
    """Return a process-wide default cache instance."""
    global _default_cache
    if _default_cache is None:
        settings = settings or get_settings()
        _default_cache = ResultCache(settings.cache_path)
    return _default_cache
