"""End-to-end regression tests for the three known Form 283 examples.

Strategy
--------
OCR is deterministic for a given PDF (Azure DI always returns the same
output for the same bytes), so we cache the OCR result on first run.
Subsequent runs skip the Document Intelligence call and go straight to
the LLM extraction step, making the tests faster and cheaper.

Cache files live at ``tests/form_extraction/fixtures/ocr_cache/<stem>.json``
and contain at minimum ``{"markdown": "..."}``. If the cache file is
absent the test calls ``run_ocr()`` which requires live Azure Document
Intelligence credentials (``RUN_AZURE_TESTS=1``). The LLM extraction
step always requires live Azure OpenAI credentials.

Run all integration tests:
    RUN_AZURE_TESTS=1 pytest tests/form_extraction/test_e2e.py -v

Warm the OCR cache only (no LLM calls):
    RUN_AZURE_TESTS=1 pytest tests/form_extraction/test_e2e.py -v -k "ocr_cache"

Expected-value policy
---------------------
``_EXPECTED`` lists only the fields we can reliably recover from OCR on
a principled, general extractor — names, ID, direct-adjacency checkboxes,
clearly-typed text fields, clean date boxes. Fields whose OCR output is
genuinely ambiguous on a given sample (e.g. a Section 5 checkbox whose
adjacent label is broken by reading-order noise) are intentionally left
out. A principled extractor is allowed to emit ``""`` for those, and the
test does not punish it for doing so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from form_extraction.core.extractor import extract
from form_extraction.core.ocr import OCRResult, run_ocr
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
_CACHE_DIR = Path(__file__).parent / "fixtures" / "ocr_cache"


# ---------------------------------------------------------------------------
# Expected field values per example (only reliably-recoverable fields)
# ---------------------------------------------------------------------------

_EXPECTED: dict[str, dict[str, Any]] = {
    "283_ex1": {
        "lastName": "טננהוים",
        "firstName": "יהודה",
        # 10 digits — non-standard; the validator flags the length.
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
            # Clean ☒ directly adjacent to מאוחדת in the OCR.
            "healthFundMember": "מאוחדת",
            "natureOfAccident": "",
            "medicalDiagnoses": "",
        },
    },
    "283_ex2": {
        "lastName": "הלוי",
        "firstName": "שלמה",
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
        "timeOfInjury": "12:00",
        "accidentLocation": "במפעל",
        "accidentAddress": "האופים 17 בני ברק",
        "accidentDescription": "במהלך העבודה נשרף ממגש לוהט.",
        "injuredBodyPart": "הפנים במיוחד הלחי הימנית",
        "signature": "",
        "formFillingDate": {"day": "14", "month": "09", "year": "2006"},
        "formReceiptDateAtClinic": {"day": "03", "month": "07", "year": "2001"},
        # ``dateOfInjury`` and ``medicalInstitutionFields.healthFundMember``
        # intentionally omitted: the DDMMYYYY box for the date is broken by
        # a stray leader glyph in the OCR and the Section 5 fund row is
        # fragmented (floating ☒ + one fund label missing its ☐). A
        # principled extractor can legitimately emit ``""`` for both.
    },
    "283_ex3": {
        "lastName": "יוחננוף",
        "firstName": "רועי",
        # 10 digits — non-standard; the validator flags the length.
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
        "signature": "רועי",
        "formFillingDate": {"day": "20", "month": "05", "year": "1999"},
        "formReceiptDateAtClinic": {"day": "30", "month": "06", "year": "1999"},
        "medicalInstitutionFields": {
            # Section 5 is fragmented: ☐ is paired with כללית inside the
            # fund-row, the ☒ floats alone, and the other fund names
            # appear after the accident-nature heading. A principled
            # extractor emits "" rather than guessing.
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
    return TEST_DATA / f"{stem}.pdf"


def _cache_path(stem: str) -> Path:
    return _CACHE_DIR / f"{stem}.json"


def _get_ocr(stem: str) -> OCRResult:
    """Return cached OCR result, generating and caching it if absent.

    First call for a given stem requires Azure Document Intelligence.
    All subsequent calls are local reads. Older cache files may carry
    extra keys (e.g. ``coord_hints`` from a previous pipeline revision);
    we ignore them.
    """
    cache = _cache_path(stem)
    if cache.exists() and cache.stat().st_size > 0:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return OCRResult(markdown=payload["markdown"])

    pdf = _pdf_path(stem)
    if not pdf.exists():
        pytest.skip(f"PDF not found: {pdf}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    ocr = run_ocr(pdf.read_bytes())
    cache.write_text(
        json.dumps({"markdown": ocr.markdown}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ocr


# ---------------------------------------------------------------------------
# Parametrised test cases
# ---------------------------------------------------------------------------

_CASES = ["283_ex1", "283_ex2", "283_ex3"]


@pytest.mark.integration
@pytest.mark.parametrize("stem", _CASES)
def test_ocr_cache(stem: str) -> None:
    """Ensure OCR cache exists (or create it). Fast no-op if already cached."""
    ocr = _get_ocr(stem)
    assert ocr.markdown, f"OCR returned empty markdown for {stem}"
    # Shape check: DI Markdown always contains at least one table row or
    # heading for a form this dense. Keep deliberately loose.
    assert any(marker in ocr.markdown for marker in ("|", "#")), (
        "Markdown lacks any table or heading markers — did DI return plain text?"
    )


@pytest.mark.integration
@pytest.mark.parametrize("stem", _CASES)
def test_extraction_fields(stem: str) -> None:
    """Run extraction on cached OCR and assert field-level correctness."""
    ocr = _get_ocr(stem)

    form = extract(ocr.markdown)
    report = validate(form, ocr_text=ocr.markdown)
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

    # Completeness must be reasonable (at least 50% of fields filled).
    assert report.completeness >= 0.5, (
        f"{stem}: completeness={report.completeness:.0%} — too many empty fields."
    )

    # No hallucination warnings on critical free-text fields.
    hallucination_issues = [
        i for i in report.issues
        if "hallucination" in i.message.lower()
        and i.field in {"lastName", "firstName", "jobType"}
    ]
    assert not hallucination_issues, (
        f"{stem}: possible hallucination in critical fields: {hallucination_issues}"
    )

    # Checkbox enum values must come from the allowed sets. Pydantic
    # already enforces this, but asserting here surfaces a clearer
    # message than a schema error.
    valid_genders = {*GENDER_LABELS, ""}
    valid_locations = {*ACCIDENT_LOCATION_LABELS, ""}
    valid_funds = {*HEALTH_FUND_LABELS, ""}
    assert data["gender"] in valid_genders, f"{stem}: invalid gender {data['gender']!r}"
    assert data["accidentLocation"] in valid_locations, (
        f"{stem}: invalid accidentLocation {data['accidentLocation']!r}"
    )
    assert data["medicalInstitutionFields"]["healthFundMember"] in valid_funds, (
        f"{stem}: invalid healthFundMember "
        f"{data['medicalInstitutionFields']['healthFundMember']!r}"
    )
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
    ocr = _get_ocr(stem)
    form = extract(ocr.markdown)
    report = validate(form, ocr_text=ocr.markdown)

    error_fields = {i.field for i in report.issues if i.severity == "error"}

    if stem == "283_ex1":
        # ex1 has a 10-digit idNumber — must be flagged.
        assert "idNumber" in error_fields, (
            "ex1: 10-digit idNumber should trigger a validation error"
        )

    if stem == "283_ex2":
        # ex2 has a valid 9-digit idNumber.
        assert "idNumber" not in error_fields, (
            f"ex2: idNumber should be valid 9 digits but got errors: "
            f"{[i for i in report.issues if i.field == 'idNumber']}"
        )

    if stem == "283_ex3":
        # ex3 also has a 10-digit idNumber — must be flagged.
        assert "idNumber" in error_fields, (
            "ex3: 10-digit idNumber should trigger a validation error"
        )

    # No date errors on any example: dates live in fixed digit boxes the
    # LLM reads cleanly and splits into DDMMYYYY parts.
    date_errors = [
        i for i in report.issues
        if i.severity == "error" and "date" in i.field.lower()
    ]
    assert not date_errors, f"{stem}: unexpected date errors: {date_errors}"
