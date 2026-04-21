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
    log.info("pipeline.start bytes=%d", len(data))
    t0 = time.perf_counter()
    ocr = run_ocr(data)
    form = extract(ocr.markdown)
    report = validate(form, ocr_text=ocr.markdown)
    log.info(
        "pipeline.done filled=%d/%d issues=%d elapsed_ms=%d",
        report.filled, report.total, len(report.issues),
        int((time.perf_counter() - t0) * 1000),
    )
    return PipelineResult(form=form, report=report, ocr_text=ocr.markdown)
