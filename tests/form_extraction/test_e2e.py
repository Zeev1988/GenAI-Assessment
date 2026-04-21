"""End-to-end smoke test for the Form 283 extractor.

Requires live Azure Document Intelligence + OpenAI credentials.
Skipped by default unless RUN_AZURE_TESTS=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from form_extraction.core.extractor import extract
from form_extraction.core.ocr import OCRResult, run_ocr

TEST_DATA = Path(__file__).parent / "test_data"
_CACHE_DIR = Path(__file__).parent / "fixtures" / "ocr_cache"


@pytest.fixture(autouse=True)
def _require_azure() -> None:
    if os.getenv("RUN_AZURE_TESTS") != "1":
        pytest.skip("RUN_AZURE_TESTS != 1")


def _get_ocr(stem: str) -> OCRResult:
    cache = _CACHE_DIR / f"{stem}.json"
    if cache.exists() and cache.stat().st_size > 0:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return OCRResult(markdown=payload["markdown"])

    pdf = TEST_DATA / f"{stem}.pdf"
    if not pdf.exists():
        pytest.skip(f"PDF not found: {pdf}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    ocr = run_ocr(pdf.read_bytes())
    cache.write_text(
        json.dumps({"markdown": ocr.markdown}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ocr


def test_extraction_smoke() -> None:
    """Run OCR + extraction on one real form and check the key fields come back."""
    ocr = _get_ocr("283_ex1")
    form = extract(ocr.markdown)
    data = form.model_dump()

    assert data["lastName"] == "טננהוים"
    assert data["firstName"] == "יהודה"
    assert data["idNumber"] == "8775245631"
    assert data["gender"] == "זכר"
