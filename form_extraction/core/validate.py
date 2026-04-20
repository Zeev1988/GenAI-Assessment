"""Post-extraction validation and completeness scoring.

The extractor returns a schema-valid ``ExtractedForm`` (guaranteed by
``response_format=json_schema`` in strict mode plus Pydantic). This
module layers two observational checks on top:

  * **Format rules.** ID length, phone shape, HH:MM time, real
    calendar dates. These are the few constraints the schema itself
    cannot express.
  * **Grounding.** Free-text values that do not appear as substrings
    of the OCR Markdown are flagged as possible hallucinations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from form_extraction.core.digits import parse_numeric, warn_only_fields
from form_extraction.core.schemas import DatePart, ExtractedForm

# Fields in the digit registry whose parser result belongs to the
# nested ``Address`` sub-model rather than the top-level form. Used
# to look up the LLM's value when comparing against the parser's
# reading for the warn-only check.
_ADDRESS_FIELDS: frozenset[str] = frozenset(
    {"street", "houseNumber", "entrance", "apartment", "city", "postalCode", "poBox"}
)

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
# Israeli landlines: 9 digits, area codes 02/03/04/08/09.
_LANDLINE_RE = re.compile(r"^0[234689]\d{7}$")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

# We strip punctuation and whitespace before grounding comparison; OCR
# frequently spaces Hebrew text differently than the extracted value, so
# we keep only letters and digits — the characters that carry meaning.
_GROUND_STRIP_RE = re.compile(r"[^\w]", flags=re.UNICODE)

# Free-text fields worth grounding. Structured/transformed fields (IDs,
# phones, dates, times) are covered by format validators; enum fields
# come from checkbox resolution rather than from the OCR body, so
# substring grounding would produce false-positive hallucination
# warnings on them.
_GROUNDED_FIELDS: tuple[str, ...] = (
    "lastName",
    "firstName",
    "jobType",
    "accidentAddress",
    "accidentDescription",
    "injuredBodyPart",
)
_GROUNDED_ADDRESS_FIELDS: tuple[str, ...] = ("street", "city")
# Short values produce too many false positives (e.g. "X" appears in
# many OCR contexts). Skip anything under this length.
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


def _check_anchor_disagreement(
    form: ExtractedForm, ocr_text: str
) -> list[Issue]:
    """Warn when the anchor-label parser reads a different value than the LLM.

    This runs for fields the extractor does *not* silently override
    because their structural check is too weak to trust over the LLM
    (apartment, entrance, houseNumber). The parser's read is still
    informative: when it disagrees with the LLM, something is off —
    either the anchor window grabbed the wrong digit run, or the
    LLM misread the box. We surface the mismatch as a warning and
    leave the JSON alone; the reviewer decides which value is right.
    """
    issues: list[Issue] = []
    for field in warn_only_fields():
        parsed = parse_numeric(ocr_text, field)
        if parsed is None:
            continue
        if field in _ADDRESS_FIELDS:
            llm_value = getattr(form.address, field, "")
            path = f"address.{field}"
        else:
            llm_value = getattr(form, field, "")
            path = field
        if not llm_value:
            # Parser read something, LLM left it empty — worth flagging
            # so the reviewer can look, but without asserting the read
            # is right.
            issues.append(
                Issue(
                    path,
                    "warning",
                    f"Anchor-label parser read '{parsed}' but the extractor left this field empty.",
                )
            )
        elif llm_value != parsed:
            issues.append(
                Issue(
                    path,
                    "warning",
                    f"Anchor-label parser read '{parsed}' but the extractor returned '{llm_value}'.",
                )
            )
    return issues


def _check_grounding(form: ExtractedForm, ocr_text: str) -> list[Issue]:
    """Flag free-text values that don't appear as substrings of the OCR."""
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
        issues.extend(_check_anchor_disagreement(form, ocr_text))
        issues.extend(_check_grounding(form, ocr_text))

    data = form.model_dump()
    filled, total = _count(data)
    completeness = filled / total if total else 0.0
    return ValidationReport(completeness=completeness, filled=filled, total=total, issues=issues)
