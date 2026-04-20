"""Gated end-to-end test against live Azure services.

Runs only when RUN_AZURE_TESTS=1 and a sample PDF is available in
../phase1_data/. Exercises OCR + LLM + validation together.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from form_extraction.core import run

pytestmark = pytest.mark.integration

TEST_DIR = Path(__file__).resolve().parents[0]
DATA_DIR = TEST_DIR / "test_data"


def _sample_pdfs() -> list[Path]:
    return sorted(DATA_DIR.glob("283_ex*.pdf")) if DATA_DIR.is_dir() else []


@pytest.fixture(autouse=True)
def _skip_unless_enabled() -> None:
    if os.getenv("RUN_AZURE_TESTS") != "1":
        pytest.skip("RUN_AZURE_TESTS != 1")


def test_pipeline_on_first_sample() -> None:
    pdfs = _sample_pdfs()
    if not pdfs:
        pytest.skip(f"No 283_ex*.pdf samples found in {DATA_DIR}")

    result = run(pdfs[0].read_bytes())
    assert result.ocr_text, "OCR produced no text"
    assert result.report.completeness > 0.2, (
        f"Completeness suspiciously low: {result.report.completeness:.2%}"
    )
    # The extracted payload must round-trip through the schema.
    assert result.form.model_dump()["lastName"] is not None
