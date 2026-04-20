"""Validation rules + completeness count."""

from __future__ import annotations

from form_extraction.core.schemas import DatePart, ExtractedForm
from form_extraction.core.validate import validate


def _form(**overrides: object) -> ExtractedForm:
    return ExtractedForm.model_validate({**ExtractedForm().model_dump(), **overrides})


def test_empty_form_has_no_issues_and_zero_completeness() -> None:
    report = validate(ExtractedForm())
    assert report.issues == []
    assert report.completeness == 0.0


def test_valid_id_and_phones_produce_no_issues() -> None:
    report = validate(_form(idNumber="123456789", mobilePhone="0501234567", landlinePhone="031234567"))
    assert report.issues == []


def test_bad_id_length_is_flagged_as_error() -> None:
    report = validate(_form(idNumber="12345"))
    assert any(i.field == "idNumber" and i.severity == "error" for i in report.issues)


def test_mobile_phone_not_starting_with_05_is_flagged() -> None:
    report = validate(_form(mobilePhone="0312345678"))
    assert any(i.field == "mobilePhone" and i.severity == "error" for i in report.issues)


def test_bad_time_format_is_flagged() -> None:
    report = validate(_form(timeOfInjury="9am"))
    assert any(i.field == "timeOfInjury" for i in report.issues)


def test_partial_date_is_flagged() -> None:
    report = validate(_form(dateOfBirth=DatePart(day="01", month="", year="1990").model_dump()))
    assert any(i.field == "dateOfBirth" for i in report.issues)


def test_invalid_calendar_date_is_flagged() -> None:
    report = validate(_form(dateOfBirth=DatePart(day="31", month="02", year="1990").model_dump()))
    assert any(i.field == "dateOfBirth" for i in report.issues)


def test_completeness_counts_filled_leaves() -> None:
    report = validate(_form(lastName="Cohen", firstName="Dana"))
    assert report.filled == 2
    assert 0 < report.completeness < 1


def test_grounding_passes_when_values_appear_in_ocr() -> None:
    ocr = "שם משפחה: כהן   שם פרטי: דנה\nסוג העבודה: מהנדסת תוכנה"
    report = validate(
        _form(lastName="כהן", firstName="דנה", jobType="מהנדסת תוכנה"),
        ocr_text=ocr,
    )
    assert all(i.message != "Value does not appear in the OCR text; possible hallucination."
               for i in report.issues)


def test_grounding_flags_hallucinated_free_text_field() -> None:
    # jobType is a grounded free-text field. If the model invents a value
    # that isn't on the page, the grounding check should flag it.
    ocr = "שם משפחה: לוי   שם פרטי: רועי\nסוג העבודה: ________________"
    report = validate(
        _form(lastName="לוי", firstName="רועי", jobType="מהנדס תוכנה"),
        ocr_text=ocr,
    )
    flagged = {i.field for i in report.issues if "hallucination" in i.message}
    assert "jobType" in flagged


def test_grounding_is_whitespace_and_punctuation_insensitive() -> None:
    # OCR says "Dana Cohen"; extracted jobType is "DanaCohen".
    # Normalization should collapse to the same form → no warning.
    ocr = "סוג העבודה: Dana Cohen"
    report = validate(_form(jobType="DanaCohen"), ocr_text=ocr)
    assert not any("hallucination" in i.message for i in report.issues)


def test_grounding_ignores_short_values() -> None:
    # Two-letter values cause too many false positives; we skip them.
    ocr = "something entirely unrelated"
    report = validate(_form(firstName="AB"), ocr_text=ocr)
    assert not any("hallucination" in i.message for i in report.issues)


def test_grounding_skipped_when_no_ocr_provided() -> None:
    # Callers without OCR text get format checks only, no grounding.
    report = validate(_form(lastName="ghost"))
    assert not any("hallucination" in i.message for i in report.issues)
