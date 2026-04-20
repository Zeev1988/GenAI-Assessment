"""End-to-end pipeline: OCR → extract → validate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from form_extraction.core.extractor import extract
from form_extraction.core.ocr import run_ocr
from form_extraction.core.schemas import ExtractedForm
from form_extraction.core.validate import ValidationReport, validate

log = logging.getLogger("form_extraction.pipeline")


@dataclass
class PipelineResult:
    form: ExtractedForm
    report: ValidationReport
    ocr_text: str


def run(data: bytes) -> PipelineResult:
    """Run OCR, extract fields, validate. Returns the form plus a report."""
    log.info("pipeline.start bytes=%d", len(data))
    t0 = time.perf_counter()
    ocr_text = run_ocr(data)
    form = extract(ocr_text)
    # Pass ocr_text so the validator can run the grounding check that flags
    # free-text values missing from the page (likely hallucinations).
    report = validate(form, ocr_text=ocr_text)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "pipeline.done filled=%d total=%d issues=%d elapsed_ms=%d",
        report.filled, report.total, len(report.issues), elapsed_ms,
    )
    if report.issues:
        for issue in report.issues:
            log.info(
                "pipeline.issue field=%s severity=%s msg=%s",
                issue.field, issue.severity, issue.message,
            )
    _log_extracted_fields_debug(form)
    return PipelineResult(form=form, report=report, ocr_text=ocr_text)


def _log_extracted_fields_debug(form: ExtractedForm) -> None:
    """DEBUG-only per-field log. PII-safe by default — enable DEBUG to see."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    d = form.model_dump()

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, str):
            if node.strip():
                log.debug("pipeline.field %-45s = %r", path, node)
            else:
                log.debug("pipeline.field %-45s = <empty>", path)

    _walk(d, "")
