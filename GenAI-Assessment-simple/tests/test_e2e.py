"""End-to-end regression tests for the three known Form 283 examples.

Strategy
--------
OCR is deterministic for a given PDF (Azure DI always returns the same
Markdown for the same bytes), so we cache the OCR output on first run.
Subsequent runs skip the Document Intelligence call and go straight to the
LLM extraction step, making the tests faster and cheaper.

Cache files live at  tests/fixtures/ocr_cache/<stem>.txt
If the cache file is absent the test calls run_ocr() which requires live
Azure Document Intelligence credentials  (RUN_AZURE_TESTS=1).
The LLM extraction step always requires live Azure OpenAI credentials.

Run all integration tests:
    RUN_AZURE_TESTS=1 pytest tests/test_e2e.py -v

Warm the OCR cache only (no LLM calls):
    RUN_AZURE_TESTS=1 pytest tests/test_e2e.py -v -k "ocr_cache"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from form_extraction.core.extractor import extract
from form_extraction.core.ocr import run_ocr
from form_extraction.core.validate import validate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA = Path(__file__).parent / "test_data"
_PHASE1 = next(
    (p for p in (TEST_DATA / "phase1_data", TEST_DATA.parent / "phase1_data") if p.is_dir()),
    TEST_DATA / "phase1_data",  # fallback – will be created on first cache write
)
_CACHE_DIR = Path(__file__).parent / "fixtures" / "ocr_cache"


# ---------------------------------------------------------------------------
# Expected field values for each example
# ---------------------------------------------------------------------------
# Keys map to ExtractedForm field names.  Only the fields we are confident
# about are listed; the test asserts these exactly and ignores the rest.
# Fields not listed here are covered by the structural assertions below.
#
# To add or update expected values after a new verified run, edit these dicts.
# ---------------------------------------------------------------------------

_EXPECTED: dict[str, dict[str, Any]] = {
    "283_ex1": {
        # TODO: fill in after first verified run.
        # Add entries like: "lastName": "...", "idNumber": "...", etc.
    },
    "283_ex2": {
        "lastName": "הלוי",
        "firstName": "שלמה",
        # idNumber resolved via Python normalisation (no Hebrew prefix on ex2):
        # OCR: "02245612 0" → strip → "022456120" (9 digits, valid)
        "idNumber": "022456120",
        "gender": "זכר",
        "dateOfBirth": {"day": "14", "month": "10", "year": "1990"},
        "address": {
            "street": "חיים ויצמן",
            "houseNumber": "6",
            "entrance": "",
            "apartment": "34",
            "city": "יוקנעם",
            "postalCode": "4454124",
            "poBox": "",
        },
        "dateOfInjury": {"day": "12", "month": "08", "year": "2005"},
        "timeOfInjury": "12:00",
        "accidentLocation": "במפעל",
        "accidentAddress": "האופים 17 בני ברק",
        "accidentDescription": "במהלך העבודה נשרף ממגש לוהט.",
        "injuredBodyPart": "הפנים במיוחד הלחי הימנית",
        "signature": "שלמה הלוי",
        "formFillingDate": {"day": "14", "month": "09", "year": "2006"},
        "formReceiptDateAtClinic": {"day": "03", "month": "07", "year": "2001"},
        "medicalInstitutionFields": {
            # כללית checkbox was spatially selected by Azure DI
            "healthFundMember": "כללית",
            "natureOfAccident": "",
            "medicalDiagnoses": "",
        },
    },
    "283_ex3": {
        "lastName": "יוחננוף",
        "firstName": "רועי",
        # idNumber resolved via Python normalisation (Hebrew prefix "עי" present):
        # OCR: "עי 7 6 5 1| 2 5 | 4 3 3 | 0" → strip → "7651254330" → reverse →
        # "0334521567" (10 digits — validator will flag; no trimming)
        "idNumber": "0334521567",
        "gender": "זכר",
        "dateOfBirth": {"day": "03", "month": "03", "year": "1974"},
        "address": {
            "street": "המאיר",
            "houseNumber": "15",
            "entrance": "1",
            "apartment": "16",
            "city": "אלוני הבשן",
            "postalCode": "445412",
            "poBox": "",
        },
        "jobType": "ירקנייה",
        "dateOfInjury": {"day": "14", "month": "04", "year": "1999"},
        "timeOfInjury": "15:30",
        "accidentLocation": "במפעל",
        "accidentAddress": "לוונברג 173 כפר סבא",
        "accidentDescription": "במהלך העבודה הרמתי משקל כבד וכתוצאה מכך הייתי צריך ניתוח קילה",
        "injuredBodyPart": "קילה",
        "signature": "רועי יוחננוף",
        "formFillingDate": {"day": "20", "month": "05", "year": "1999"},
        "formReceiptDateAtClinic": {"day": "30", "month": "06", "year": "1999"},
        "medicalInstitutionFields": {
            # Only membership-status checkbox was marked; no fund-name checkbox.
            # NORMALIZED_HEALTH_FUND emits "" → LLM must copy verbatim.
            "healthFundMember": "",
            "natureOfAccident": "",
            "medicalDiagnoses": "",
        },
    },
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _require_azure() -> None:
    """Skip the whole module unless RUN_AZURE_TESTS=1."""
    if os.getenv("RUN_AZURE_TESTS") != "1":
        pytest.skip("RUN_AZURE_TESTS != 1")


def _pdf_path(stem: str) -> Path:
    return _PHASE1 / f"{stem}.pdf"


def _cache_path(stem: str) -> Path:
    return _CACHE_DIR / f"{stem}.txt"


def _get_ocr(stem: str) -> str:
    """Return cached OCR text, generating and caching it if absent.

    First call for a given stem requires Azure Document Intelligence.
    All subsequent calls are local reads.
    """
    cache = _cache_path(stem)
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text(encoding="utf-8")

    pdf = _pdf_path(stem)
    if not pdf.exists():
        pytest.skip(f"PDF not found: {pdf}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    ocr_text = run_ocr(pdf.read_bytes())
    cache.write_text(ocr_text, encoding="utf-8")
    return ocr_text


# ---------------------------------------------------------------------------
# Parametrised test cases
# ---------------------------------------------------------------------------

_CASES = ["283_ex1", "283_ex2", "283_ex3"]


@pytest.mark.integration
@pytest.mark.parametrize("stem", _CASES)
def test_ocr_cache(stem: str) -> None:
    """Ensure OCR cache exists (or create it).  Fast no-op if already cached."""
    ocr_text = _get_ocr(stem)
    assert ocr_text, f"OCR returned empty text for {stem}"
    # Sanity: the pre-processed OCR should always contain our spatial header
    # and the Markdown form body.
    assert "=== FORM 283 SPATIAL EXTRACTION ===" in ocr_text, (
        "Spatial extraction header missing"
    )
    assert "=== FORM BODY (markdown OCR) ===" in ocr_text, (
        "Markdown form body header missing"
    )
    # The spatial header always lists these pre-computed field rows.
    for key in ("formReceiptDateAtClinic:", "formFillingDate:", "dateOfInjury:",
                "idNumber:", "mobilePhone:", "healthFundMember (resolved):"):
        assert key in ocr_text, f"Spatial header is missing field row: {key!r}"


@pytest.mark.integration
@pytest.mark.parametrize("stem", _CASES)
def test_extraction_fields(stem: str) -> None:
    """Run extraction on cached OCR and assert field-level correctness."""
    ocr_text = _get_ocr(stem)

    form = extract(ocr_text)
    report = validate(form, ocr_text=ocr_text)
    data = form.model_dump()

    # --- Structural assertions (all examples) --------------------------------

    # Every leaf must be a string (no None, no unexpected types)
    def _check_leaves(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _check_leaves(v, f"{path}.{k}" if path else k)
        elif not isinstance(node, str):
            pytest.fail(f"Non-string leaf at {path!r}: {node!r}")

    _check_leaves(data)

    # Completeness must be reasonable (at least 50% of fields filled)
    assert report.completeness >= 0.5, (
        f"{stem}: completeness={report.completeness:.0%} — too many empty fields.\n"
        f"Empty fields: {[i.field for i in report.issues if 'empty' in i.message.lower()]}"
    )

    # No hallucination warnings on checkbox or name fields (grounding check)
    hallucination_issues = [
        i for i in report.issues
        if "hallucination" in i.message.lower()
        and i.field in {"lastName", "firstName", "accidentLocation", "gender"}
    ]
    assert not hallucination_issues, (
        f"{stem}: possible hallucination in critical fields: {hallucination_issues}"
    )

    # Checkbox enum values must be valid (Pydantic already enforces this, but
    # make the failure message clearer)
    valid_genders = {"זכר", "נקבה", ""}
    valid_locations = {
        "במפעל", "מחוץ למפעל", "בדרך לעבודה", "בדרך מהעבודה",
        "ת. דרכים בעבודה", "אחר", "",
    }
    valid_funds = {"כללית", "מכבי", "מאוחדת", "לאומית", ""}
    assert data["gender"] in valid_genders, f"{stem}: invalid gender {data['gender']!r}"
    assert data["accidentLocation"] in valid_locations, (
        f"{stem}: invalid accidentLocation {data['accidentLocation']!r}"
    )
    assert data["medicalInstitutionFields"]["healthFundMember"] in valid_funds, (
        f"{stem}: invalid healthFundMember {data['medicalInstitutionFields']['healthFundMember']!r}"
    )

    # --- Field-level assertions for cases where expected values are defined --

    expected = _EXPECTED.get(stem, {})
    if not expected:
        # No expected values defined yet — skip field assertions
        return

    mismatches: list[str] = []

    def _assert_field(path: str, actual: Any, exp: Any) -> None:
        if isinstance(exp, dict):
            for k, v in exp.items():
                _assert_field(f"{path}.{k}", actual.get(k) if isinstance(actual, dict) else None, v)
        else:
            if actual != exp:
                mismatches.append(f"  {path}: expected={exp!r}  actual={actual!r}")

    for field, exp_value in expected.items():
        _assert_field(field, data.get(field), exp_value)

    if mismatches:
        pytest.fail(
            f"{stem}: {len(mismatches)} field mismatch(es):\n" + "\n".join(mismatches)
        )


@pytest.mark.integration
@pytest.mark.parametrize("stem", _CASES)
def test_validation_issues_match_expectations(stem: str) -> None:
    """Assert that known validation issues are present/absent as expected."""
    ocr_text = _get_ocr(stem)
    form = extract(ocr_text)
    report = validate(form, ocr_text=ocr_text)

    issue_fields = {i.field for i in report.issues}
    error_fields = {i.field for i in report.issues if i.severity == "error"}

    if stem == "283_ex2":
        # Phones are garbled by OCR (digit-box format confusion)
        # but no critical field errors expected
        assert "idNumber" not in error_fields, (
            f"ex2: idNumber should be valid 9 digits but got errors: "
            f"{[i for i in report.issues if i.field == 'idNumber']}"
        )

    if stem == "283_ex3":
        # ID is 10 digits (a real data quality issue on this form) — must be flagged
        assert "idNumber" in error_fields, (
            "ex3: 10-digit idNumber should trigger a validation error but did not"
        )

    # No date errors expected on any example (dates come from digit boxes
    # which our preprocessing normalises)
    date_errors = [
        i for i in report.issues
        if i.severity == "error" and "date" in i.field.lower()
    ]
    assert not date_errors, f"{stem}: unexpected date errors: {date_errors}"
