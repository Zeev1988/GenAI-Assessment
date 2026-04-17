"""End-to-end async pipeline: OCR -> LLM extraction -> validation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from common.cache import ResultCache, get_cache
from common.config import Settings, get_settings
from common.errors import AppError
from common.logging_config import correlation_scope, get_logger
from common.security import validate_upload

from form_extraction.backend.extraction.extractor import extract_fields
from form_extraction.backend.ocr.layout import OCRResult, run_ocr
from form_extraction.backend.schemas import ExtractedForm
from form_extraction.backend.validation.judge import judge_extraction
from form_extraction.backend.validation.report import ValidationReport, build_report

_log = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    form: ExtractedForm
    report: ValidationReport
    ocr: OCRResult
    fingerprint: str
    mime: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    from_cache: dict[str, bool] = field(default_factory=dict)
    correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "fingerprint": self.fingerprint,
            "mime": self.mime,
            "extracted": self.form.model_dump(),
            "validation": self.report.to_dict(),
            "ocr": {
                "language_hint": self.ocr.language_hint,
                "page_count": self.ocr.page_count,
                "has_tables": self.ocr.has_tables,
                "char_count": len(self.ocr.content),
            },
            "stage_timings_ms": {k: round(v, 2) for k, v in self.stage_timings_ms.items()},
            "from_cache": dict(self.from_cache),
        }


async def run_pipeline(
    data: bytes,
    filename: str | None = None,
    *,
    settings: Settings | None = None,
    cache: ResultCache | None = None,
    correlation_id: str | None = None,
    force_refresh: bool = False,
) -> PipelineResult:
    """Run OCR, extraction, and validation on ``data``."""
    settings = settings or get_settings()
    cache = cache or get_cache(settings)

    with correlation_scope(correlation_id) as cid:
        mime = validate_upload(data, filename, settings.max_upload_bytes)
        fingerprint = ResultCache.fingerprint(data)

        timings: dict[str, float] = {}
        from_cache: dict[str, bool] = {"ocr": False, "extract": False, "judge": False}

        _log.info(
            "pipeline.start",
            filename=filename,
            mime=mime,
            size_bytes=len(data),
            fingerprint=fingerprint[:12],
            enable_judge=settings.app_enable_llm_judge,
        )

        # --- OCR --------------------------------------------------------
        t0 = time.perf_counter()
        ocr: OCRResult | None = None if force_refresh else cache.get("ocr", fingerprint)
        if ocr is None:
            ocr = await run_ocr(data, settings=settings)
            cache.set("ocr", fingerprint, ocr)
        else:
            from_cache["ocr"] = True
        timings["ocr_ms"] = (time.perf_counter() - t0) * 1000.0

        # --- Extraction -------------------------------------------------
        t1 = time.perf_counter()
        cached_extract = None if force_refresh else cache.get("extract", fingerprint)
        if cached_extract is None:
            form = await extract_fields(ocr, settings=settings)
            cache.set("extract", fingerprint, form.model_dump())
        else:
            form = ExtractedForm.model_validate(cached_extract)
            from_cache["extract"] = True
        timings["extract_ms"] = (time.perf_counter() - t1) * 1000.0

        # --- Validation -------------------------------------------------
        t2 = time.perf_counter()
        report = build_report(form)
        timings["validate_ms"] = (time.perf_counter() - t2) * 1000.0

        if settings.app_enable_llm_judge:
            t3 = time.perf_counter()
            score, comments = await judge_extraction(ocr.content, form, settings=settings)
            report.judge_score = score
            report.judge_comments = comments
            timings["judge_ms"] = (time.perf_counter() - t3) * 1000.0

        total_ms = sum(timings.values())
        _log.info(
            "pipeline.complete",
            total_ms=round(total_ms, 2),
            completeness=round(report.completeness, 4),
            issues=len(report.issues),
            from_cache=from_cache,
        )

        return PipelineResult(
            form=form,
            report=report,
            ocr=ocr,
            fingerprint=fingerprint,
            mime=mime,
            stage_timings_ms=timings,
            from_cache=from_cache,
            correlation_id=cid,
        )


def run_pipeline_sync(
    data: bytes,
    filename: str | None = None,
    **kwargs: Any,
) -> PipelineResult:
    """Synchronous wrapper for environments like Streamlit."""
    try:
        return asyncio.run(run_pipeline(data, filename, **kwargs))
    except AppError:
        raise
    except Exception as exc:
        _log.exception("pipeline.unexpected_error", error=str(exc))
        raise
