"""Simple accuracy + completeness validation for an ExtractedForm.

Accuracy here means "the values the LLM returned are in the format we expect
for their field" — ID is 9 digits, phones start with 0, times are HH:MM,
dates are real calendar dates. We deliberately stop there: any check that
tries to reason about whether a field is semantically right (e.g. which
checkbox the form marked) belongs in the LLM prompt, not in a rules engine.

Completeness is a simple filled/total leaf count.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from form_extraction.core.schemas import DatePart, ExtractedForm

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    field: str
    severity: Severity
    message: str


@dataclass
class ValidationReport:
    completeness: float
    filled: int
    total: int
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": round(self.completeness, 4),
            "filled": self.filled,
            "total": self.total,
            "issues": [asdict(i) for i in self.issues],
        }


_ID_RE = re.compile(r"^\d{9}$")
_MOBILE_RE = re.compile(r"^05\d{8}$")
_LANDLINE_RE = re.compile(r"^0[2-489]\d{7}$")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

# Punctuation/whitespace we strip before grounding comparison. The OCR
# frequently spaces Hebrew text differently than the extracted value and
# sprinkles formatting characters the LLM collapses. We keep only letters
# and digits (any script), which is what actually carries meaning.
_GROUND_STRIP_RE = re.compile(r"[^\w]", flags=re.UNICODE)

# Free-text fields worth grounding. We exclude structured/transformed fields
# (IDs, phones, dates, times) because their format validators already catch
# bogus values and because the model legitimately rewrites them (e.g. strips
# dashes from phones) which would cause spurious grounding failures.
_GROUNDED_FIELDS: tuple[str, ...] = (
    "lastName",
    "firstName",
    "gender",
    "jobType",
    "accidentLocation",
    "accidentAddress",
    "accidentDescription",
    "injuredBodyPart",
    "signature",
)
_GROUNDED_ADDRESS_FIELDS: tuple[str, ...] = ("street", "city")
_GROUNDED_MEDICAL_FIELDS: tuple[str, ...] = (
    "healthFundMember",
    "natureOfAccident",
    "medicalDiagnoses",
)
# Minimum length before we bother grounding. One- and two-character values
# produce too many false positives (e.g. "X" matches any uppercase X in OCR).
_MIN_GROUND_LEN = 3


def _parse_date(p: DatePart) -> date | None:
    if not (p.day and p.month and p.year):
        return None
    try:
        return date(int(p.year), int(p.month), int(p.day))
    except (ValueError, TypeError):
        return None


def _count(obj: Any) -> tuple[int, int]:
    """Return (filled_leaves, total_leaves) over a plain-data tree."""
    if isinstance(obj, dict):
        f = t = 0
        for v in obj.values():
            fi, ti = _count(v)
            f += fi
            t += ti
        return f, t
    if isinstance(obj, str):
        return (1 if obj.strip() else 0), 1
    return 0, 0


def _normalize_for_ground(s: str) -> str:
    return _GROUND_STRIP_RE.sub("", s).casefold()


def _check_grounding(form: ExtractedForm, ocr_text: str) -> list[Issue]:
    """Flag free-text values that don't appear as substrings of the OCR.

    This is an observational guardrail against LLM hallucination: any value
    that isn't physically on the page is suspicious. We normalize aggressively
    (strip punctuation / whitespace, casefold) to avoid false positives caused
    by the LLM trimming or re-spacing a legitimately-present value.

    Not corrective — we never blank fields; we only surface warnings so the
    human reviewer can check the OCR tab.
    """
    normalized_ocr = _normalize_for_ground(ocr_text)
    issues: list[Issue] = []

    def check(path: str, value: str) -> None:
        if not value or len(value.strip()) < _MIN_GROUND_LEN:
            return
        normalized = _normalize_for_ground(value)
        if not normalized:
            return
        if normalized not in normalized_ocr:
            issues.append(
                Issue(
                    path,
                    "warning",
                    "Value does not appear in the OCR text; possible hallucination.",
                )
            )

    data = form.model_dump()
    for f in _GROUNDED_FIELDS:
        check(f, data.get(f, ""))
    for f in _GROUNDED_ADDRESS_FIELDS:
        check(f"address.{f}", data.get("address", {}).get(f, ""))
    for f in _GROUNDED_MEDICAL_FIELDS:
        check(
            f"medicalInstitutionFields.{f}",
            data.get("medicalInstitutionFields", {}).get(f, ""),
        )
    return issues


def validate(form: ExtractedForm, ocr_text: str | None = None) -> ValidationReport:
    issues: list[Issue] = []

    if form.idNumber and not _ID_RE.match(form.idNumber):
        issues.append(Issue("idNumber", "error", "ID number must be exactly 9 digits."))

    if form.mobilePhone and not _MOBILE_RE.match(form.mobilePhone):
        issues.append(
            Issue("mobilePhone", "error", "Mobile phone must be 10 digits starting with 05.")
        )

    if form.landlinePhone and not _LANDLINE_RE.match(form.landlinePhone):
        issues.append(
            Issue(
                "landlinePhone",
                "warning",
                "Landline phone must be 9 digits starting with 02/03/04/08/09.",
            )
        )

    if form.timeOfInjury and not _TIME_RE.match(form.timeOfInjury):
        issues.append(
            Issue("timeOfInjury", "warning", "Time of injury must be HH:MM in 24-hour format.")
        )

    for name, part in (
        ("dateOfBirth", form.dateOfBirth),
        ("dateOfInjury", form.dateOfInjury),
        ("formFillingDate", form.formFillingDate),
        ("formReceiptDateAtClinic", form.formReceiptDateAtClinic),
    ):
        filled_parts = sum(1 for x in (part.day, part.month, part.year) if x)
        if 0 < filled_parts < 3:
            issues.append(
                Issue(name, "warning", "Date is partially filled (expected day + month + year).")
            )
        elif filled_parts == 3 and _parse_date(part) is None:
            issues.append(Issue(name, "warning", "Date parts do not form a valid calendar date."))

    if ocr_text:
        issues.extend(_check_grounding(form, ocr_text))

    data = form.model_dump()
    filled, total = _count(data)
    completeness = filled / total if total else 0.0
    return ValidationReport(completeness=completeness, filled=filled, total=total, issues=issues)
