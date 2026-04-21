"""Basic validation tests."""

from __future__ import annotations

from form_extraction.core.schemas import ExtractedForm
from form_extraction.core.validate import validate


def _form(**overrides: object) -> ExtractedForm:
    return ExtractedForm.model_validate({**ExtractedForm().model_dump(), **overrides})


def test_valid_id_and_phone_produce_no_issues() -> None:
    report = validate(_form(idNumber="123456789", mobilePhone="0501234567"))
    assert report.issues == []


def test_bad_id_length_is_flagged_as_error() -> None:
    report = validate(_form(idNumber="12345"))
    assert any(i.field == "idNumber" and i.severity == "error" for i in report.issues)
