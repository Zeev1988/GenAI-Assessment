"""Gated integration tests hitting the live Azure services."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from form_extraction.backend.pipeline import run_pipeline_sync

GOLDEN_DIR = Path(__file__).parent / "golden"

pytestmark = pytest.mark.integration


def _should_run() -> bool:
    return os.getenv("RUN_AZURE_TESTS") == "1"


@pytest.fixture(autouse=True)
def _skip_if_disabled() -> None:
    if not _should_run():
        pytest.skip("RUN_AZURE_TESTS != 1; skipping live Azure tests.")


def test_pipeline_on_samples(sample_pdfs: list[Path]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    update = os.getenv("UPDATE_GOLDEN") == "1"

    for pdf in sample_pdfs:
        data = pdf.read_bytes()
        result = run_pipeline_sync(data, filename=pdf.name)
        payload = result.form.model_dump()

        assert result.report.completeness >= 0.3, (
            f"Completeness too low for {pdf.name}: {result.report.completeness:.2%}"
        )

        golden_path = GOLDEN_DIR / f"{pdf.stem}.json"
        if update or not golden_path.exists():
            golden_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        assert set(expected.keys()) == set(payload.keys())
