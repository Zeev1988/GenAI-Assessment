"""Validation rules and report tests."""

from __future__ import annotations

from form_extraction.backend.schemas import DatePart, ExtractedForm
from form_extraction.backend.validation.report import build_report
from form_extraction.backend.validation.rules import validate


def _make_form(**kwargs: object) -> ExtractedForm:
    return ExtractedForm.model_validate({**ExtractedForm().model_dump(), **kwargs})


def test_empty_form_has_no_issues_but_zero_completeness() -> None:
    report = build_report(ExtractedForm())
    assert report.issues == []
    assert report.completeness == 0.0
    assert report.filled_fields == 0
    assert report.total_fields > 0


def test_valid_israeli_id_passes_checksum() -> None:
    form = _make_form(idNumber="123456782")
    issues = validate(form)
    assert all(i.field != "idNumber" for i in issues)


def test_bad_length_id_errors() -> None:
    form = _make_form(idNumber="1234")
    issues = [i for i in validate(form) if i.field == "idNumber"]
    assert any(i.severity == "error" for i in issues)


def test_failed_checksum_is_warning() -> None:
    form = _make_form(idNumber="123456789")
    issues = [i for i in validate(form) if i.field == "idNumber"]
    assert any(i.severity == "warning" for i in issues)


def test_mobile_phone_pattern() -> None:
    bad = _make_form(mobilePhone="12345")
    good = _make_form(mobilePhone="0501234567")
    assert any(i.field == "mobilePhone" for i in validate(bad))
    assert all(i.field != "mobilePhone" for i in validate(good))


def test_partial_date_is_flagged() -> None:
    form = _make_form(dateOfBirth=DatePart(day="01", month="", year="1990").model_dump())
    issues = [i for i in validate(form) if i.field == "dateOfBirth"]
    assert any(i.severity == "warning" for i in issues)


def test_cross_field_order_is_validated() -> None:
    form = _make_form(
        dateOfInjury=DatePart(day="01", month="01", year="2024").model_dump(),
        formFillingDate=DatePart(day="31", month="12", year="2023").model_dump(),
    )
    issues = {(i.field, i.severity) for i in validate(form)}
    assert ("formFillingDate", "warning") in issues


def test_report_counts_filled_fields() -> None:
    form = _make_form(lastName="Cohen", firstName="Dana")
    report = build_report(form)
    assert report.filled_fields == 2
    assert 0 < report.completeness < 1
