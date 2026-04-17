"""High-level OCR helper built on top of Azure Document Intelligence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from common.clients.doc_intelligence import analyze_layout
from common.config import Settings
from common.errors import OCRError
from common.logging_config import get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class OCRResult:
    content: str
    pages_text: list[str] = field(default_factory=list)
    page_count: int = 0
    has_tables: bool = False
    language_hint: str = "und"
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def short_summary(self) -> str:
        return (
            f"pages={self.page_count} tables={self.has_tables} "
            f"lang={self.language_hint} chars={len(self.content)}"
        )


_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _detect_language(text: str) -> str:
    """Return ``'he'``, ``'en'``, ``'mixed'``, or ``'und'`` using char counts."""
    if not text:
        return "und"
    hebrew = len(_HEBREW_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if hebrew == 0 and latin == 0:
        return "und"
    if hebrew == 0:
        return "en"
    if latin == 0:
        return "he"
    ratio = hebrew / max(hebrew + latin, 1)
    if ratio > 0.75:
        return "he"
    if ratio < 0.25:
        return "en"
    return "mixed"


def _page_text(page: Any) -> str:
    lines = getattr(page, "lines", None) or []
    return "\n".join(getattr(line, "content", "") for line in lines if getattr(line, "content", ""))


async def run_ocr(data: bytes, *, settings: Settings | None = None) -> OCRResult:
    """Run layout OCR and materialise a clean :class:`OCRResult`."""
    result = await analyze_layout(data, settings=settings)

    content = (result.content or "").strip()
    if not content:
        raise OCRError("OCR returned empty content; the document may be blank or unreadable.")

    pages = result.pages or []
    pages_text = [_page_text(p) for p in pages]
    has_tables = bool(result.tables)
    language_hint = _detect_language(content)

    metadata: dict[str, Any] = {
        "page_count": len(pages),
        "table_count": len(result.tables or []),
        "model_id": getattr(result, "model_id", None),
    }

    ocr_result = OCRResult(
        content=content,
        pages_text=pages_text,
        page_count=len(pages),
        has_tables=has_tables,
        language_hint=language_hint,
        raw_metadata=metadata,
    )

    _log.debug("ocr.summary", summary=ocr_result.short_summary())
    return ocr_result
