"""End-to-end regression tests for the three known Form 283 examples.

Strategy
--------
OCR is deterministic for a given PDF (Azure DI always returns the same
output for the same bytes), so we cache the OCR output on first run.
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

import os
from pathlib import Path
from typing import Any

import pytest

from form_extraction.core.extractor import extract
from form_extraction.core.ocr import run_ocr
from form_extraction.core.schemas import (
    ACCIDENT_LOCATION_LABELS,
    GENDER_LABELS,
    HEALTH_FUND_LABELS,
)
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
# ---------------------------------------------------------------------------

_EXPECTED: dict[str, dict[str, Any]] = {
    "283_ex1": {
        "lastName": "טננהוים",
        "firstName": "יהודה",
        # 10 digits — non-standard; validator will flag the length.
        "idNumber": "8775245631",
        "gender": "זכר",
        "dateOfBirth": {"day": "02", "month": "02", "year": "1995"},
        "address": {
            "street": "הרמבם",
            "houseNumber": "16",
            "entrance": "1",
            "apartment": "12",
            "city": "אבן יהודה",
            "postalCode": "312422",
            "poBox": "",
        },
        "jobType": "מלצרות",
        "dateOfInjury": {"day": "16", "month": "04", "year": "2022"},
        "timeOfInjury": "19:00",
        "accidentLocation": "במפעל",
        "accidentAddress": "הורדים 8, תל אביב",
        "accidentDescription": "החלקתי בגלל שהרצפה הייתה רטובה ולא היה שום שלט שמזהיר.",
        "injuredBodyPart": "יד שמאל",
        "signature": "",
        "formFillingDate": {"day": "25", "month": "01", "year": "2023"},
        "formReceiptDateAtClinic": {"day": "02", "month": "02", "year": "1999"},
        "medicalInstitutionFields": {
            "healthFundMember": "מאוחדת",
            "natureOfAccident": "",
            "medicalDiagnoses": "",
        },
    },
    "283_ex2": {
        "lastName": "הלוי",
        "firstName": "שלמה",
        # Digit-box OCR yields "02245612 0" → stripped to "022456120" (9 digits).
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
        "signature": "",
        "formFillingDate": {"day": "14", "month": "09", "year": "2006"},
        "formReceiptDateAtClinic": {"day": "03", "month": "07", "year": "2001"},
        "medicalInstitutionFields": {
            "healthFundMember": "כללית",
            "natureOfAccident": "",
            "medicalDiagnoses": "",
        },
    },
    "283_ex3": {
        "lastName": "יוחננוף",
        "firstName": "רועי",
        # 10 digits — non-standard; validator will flag the length.
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
        "signature": "",
        "formFillingDate": {"day": "20", "month": "05", "year": "1999"},
        "formReceiptDateAtClinic": {"day": "30", "month": "06", "year": "1999"},
        "medicalInstitutionFields": {
            # Only the membership-status checkbox is marked; no fund-name
            # checkbox → healthFundMember is "".
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
    assert "=== FORM 283 SPATIAL EXTRACTION ===" in ocr_text, (
        "Spatial extraction header missing"
    )
    for key in (
        "formReceiptDateAtClinic:", "formFillingDate:", "dateOfInjury:",
        "idNumber:", "mobilePhone:",
        "gender (resolved):",
        "accidentLocation (resolved):",
        "healthFundMember (resolved):",
        "lastName:", "firstName:", "city:", "jobType:", "accidentDescription:",
    ):
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
        f"{stem}: completeness={report.completeness:.0%} — too many empty fields."
    )

    # No hallucination warnings on critical free-text fields (checkbox
    # enums are no longer grounded; see validate._GROUNDED_FIELDS).
    hallucination_issues = [
        i for i in report.issues
        if "hallucination" in i.message.lower()
        and i.field in {"lastName", "firstName", "jobType"}
    ]
    assert not hallucination_issues, (
        f"{stem}: possible hallucination in critical fields: {hallucination_issues}"
    )

    # Checkbox enum values must be valid. Pydantic already enforces this,
    # but assert here so the failure message is clearer than a schema error.
    valid_genders = {*GENDER_LABELS, ""}
    valid_locations = {*ACCIDENT_LOCATION_LABELS, ""}
    valid_funds = {*HEALTH_FUND_LABELS, ""}
    assert data["gender"] in valid_genders, f"{stem}: invalid gender {data['gender']!r}"
    assert data["accidentLocation"] in valid_locations, (
        f"{stem}: invalid accidentLocation {data['accidentLocation']!r}"
    )
    assert data["medicalInstitutionFields"]["healthFundMember"] in valid_funds, (
        f"{stem}: invalid healthFundMember {data['medicalInstitutionFields']['healthFundMember']!r}"
    )

    # signature is always empty by design (see schemas.ExtractedForm).
    assert data["signature"] == "", f"{stem}: signature should always be '' but was {data['signature']!r}"

    # --- Field-level assertions against _EXPECTED ----------------------------

    expected = _EXPECTED.get(stem, {})
    if not expected:
        return

    mismatches: list[str] = []

    def _assert_field(path: str, actual: Any, exp: Any) -> None:
        if isinstance(exp, dict):
            for k, v in exp.items():
                _assert_field(f"{path}.{k}", actual.get(k) if isinstance(actual, dict) else None, v)
        else:
            if actual != exp:
                mismatches.append(f"  {path}: expected={exp!r}  actual={actual!r}")

    for field_name, exp_value in expected.items():
        _assert_field(field_name, data.get(field_name), exp_value)

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

    error_fields = {i.field for i in report.issues if i.severity == "error"}

    if stem == "283_ex1":
        # ex1 has a 10-digit idNumber — must be flagged.
        assert "idNumber" in error_fields, (
            "ex1: 10-digit idNumber should trigger a validation error"
        )

    if stem == "283_ex2":
        # ex2 has a valid 9-digit idNumber after digit-box cleanup.
        assert "idNumber" not in error_fields, (
            f"ex2: idNumber should be valid 9 digits but got errors: "
            f"{[i for i in report.issues if i.field == 'idNumber']}"
        )

    if stem == "283_ex3":
        # ex3 also has a 10-digit idNumber — must be flagged.
        assert "idNumber" in error_fields, (
            "ex3: 10-digit idNumber should trigger a validation error"
        )

    # No date errors on any example (dates come from digit boxes which our
    # preprocessing normalises into valid DDMMYYYY windows).
    date_errors = [
        i for i in report.issues
        if i.severity == "error" and "date" in i.field.lower()
    ]
    assert not date_errors, f"{stem}: unexpected date errors: {date_errors}"
