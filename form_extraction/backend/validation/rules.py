"""Rule-based validation of an :class:`ExtractedForm` instance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from form_extraction.backend.schemas import DatePart, ExtractedForm

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class Issue:
    field: str
    severity: Severity
    message: str


_ID_RE = re.compile(r"^\d{9}$")
_MOBILE_RE = re.compile(r"^05\d{8}$")
_LANDLINE_RE = re.compile(r"^0[2-489]\d{7}$")
_POSTAL_RE = re.compile(r"^\d{5}(\d{2})?$")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _is_israeli_id_valid(value: str) -> bool:
    """Teudat zehut checksum (Luhn-like, base 10, weights 1/2)."""
    if not _ID_RE.match(value):
        return False
    total = 0
    for i, ch in enumerate(value):
        digit = int(ch) * (1 if i % 2 == 0 else 2)
        total += digit if digit < 10 else digit - 9
    return total % 10 == 0


def _parse_date(part: DatePart) -> date | None:
    if not (part.day and part.month and part.year):
        return None
    try:
        return date(int(part.year), int(part.month), int(part.day))
    except (ValueError, TypeError):
        return None


def _date_is_complete(part: DatePart) -> bool:
    return bool(part.day and part.month and part.year)


def _date_is_partial(part: DatePart) -> bool:
    filled = sum(1 for v in (part.day, part.month, part.year) if v)
    return 0 < filled < 3


def validate(form: ExtractedForm) -> list[Issue]:
    issues: list[Issue] = []

    if form.idNumber:
        if not _ID_RE.match(form.idNumber):
            issues.append(Issue("idNumber", "error", "ID number must be exactly 9 digits."))
        elif not _is_israeli_id_valid(form.idNumber):
            issues.append(
                Issue(
                    "idNumber",
                    "warning",
                    "ID number fails the Israeli teudat-zehut checksum.",
                )
            )

    if form.mobilePhone and not _MOBILE_RE.match(form.mobilePhone):
        issues.append(
            Issue(
                "mobilePhone",
                "warning",
                "Mobile phone does not match the expected pattern 05XXXXXXXX.",
            )
        )

    if form.landlinePhone and not _LANDLINE_RE.match(form.landlinePhone):
        issues.append(
            Issue(
                "landlinePhone",
                "info",
                "Landline phone does not match the expected Israeli pattern.",
            )
        )

    if form.address.postalCode and not _POSTAL_RE.match(form.address.postalCode):
        issues.append(Issue("address.postalCode", "warning", "Postal code must be 5 or 7 digits."))

    if form.timeOfInjury and not _TIME_RE.match(form.timeOfInjury):
        issues.append(
            Issue("timeOfInjury", "info", "Time of injury is not in HH:MM 24-hour format.")
        )

    today = date.today()

    for field_name, part in (
        ("dateOfBirth", form.dateOfBirth),
        ("dateOfInjury", form.dateOfInjury),
        ("formFillingDate", form.formFillingDate),
        ("formReceiptDateAtClinic", form.formReceiptDateAtClinic),
    ):
        if _date_is_partial(part):
            issues.append(
                Issue(
                    field_name,
                    "warning",
                    "Date is partially filled. Expected day, month, and year together.",
                )
            )
            continue
        parsed = _parse_date(part)
        if _date_is_complete(part) and parsed is None:
            issues.append(
                Issue(field_name, "error", "Date values do not form a real calendar date.")
            )
        elif parsed and parsed > today and field_name != "formReceiptDateAtClinic":
            issues.append(Issue(field_name, "warning", "Date is in the future."))

    birth = _parse_date(form.dateOfBirth)
    injury = _parse_date(form.dateOfInjury)
    filling = _parse_date(form.formFillingDate)
    receipt = _parse_date(form.formReceiptDateAtClinic)

    if birth and injury and injury < birth:
        issues.append(Issue("dateOfInjury", "error", "Date of injury precedes date of birth."))
    if injury and filling and filling < injury:
        issues.append(
            Issue(
                "formFillingDate",
                "warning",
                "Form filling date precedes the reported injury date.",
            )
        )
    if filling and receipt and receipt < filling:
        issues.append(
            Issue(
                "formReceiptDateAtClinic",
                "warning",
                "Receipt date at clinic precedes form filling date.",
            )
        )

    return issues
