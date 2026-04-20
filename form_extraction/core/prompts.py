"""Prompt + one worked example for Form 283 extraction.

Everything the LLM sees lives here. The design is deliberately small:

* A principled ``SYSTEM_PROMPT`` that describes the form, the OCR
  rendering, and a short set of reasoning rules that apply uniformly
  to every field — rather than enumerating per-field heuristics.
* A single, clean few-shot that anchors the OCR → JSON mapping for
  the shapes that are hard to convey in prose: checkbox glyphs,
  DDMMYYYY date parts, the address table, and the empty-string
  convention for unfilled fields.
* :func:`build_messages` — assembles the system, few-shot, and real
  user turns into the chat-completions payload.

The structural contract (field names, enum values, "" defaults) is
carried by ``response_format=json_schema`` in strict mode; the prompt
does not restate it.
"""

from __future__ import annotations

import json

from form_extraction.core.schemas import (
    ACCIDENT_LOCATION_LABELS,
    GENDER_LABELS,
    HEALTH_FUND_LABELS,
)

_GENDER_LIST = " / ".join(f'"{x}"' for x in GENDER_LABELS)
_FUND_LIST = " / ".join(f'"{x}"' for x in HEALTH_FUND_LABELS)
_LOCATION_LIST = " / ".join(f'"{x}"' for x in ACCIDENT_LOCATION_LABELS)


SYSTEM_PROMPT = f"""\
You extract fields from a Bituach Leumi (ביטוח לאומי) Form 283 — Request \
for Medical Treatment for a Work Injury (Self-Employed). The form is in \
Hebrew, laid out right-to-left.

Input
-----
The user turn contains Azure Document Intelligence's Markdown rendering \
of a scanned form. Reading order is already handled. Selection marks \
appear inline as ``☒`` (checked) and ``☐`` (empty). Date boxes render as \
spaced digit runs, e.g. ``0 3 0 4 2 0 2 4``. Tables render with ``|`` \
cells. Illegible or unfilled fields appear as blank cells or missing \
content.

Output
------
A single JSON object matching the provided schema. Every property is \
required; use ``""`` for any value that is not clearly filled on the \
form. Dates are split into ``{{"day": "DD", "month": "MM", "year": \
"YYYY"}}``. The schema is enforced by the API; do not restate it, do \
not wrap the output in Markdown fences, do not add commentary.

Rules
-----
1. Copy verbatim. Preserve Hebrew characters exactly; do not translate, \
transliterate, summarise, or reformat. Digit fields are a single \
continuous string with no spaces or separators.
2. Empty stays empty. If a field is blank, illegible, or you cannot \
confidently resolve it, emit ``""``. Never guess and never copy from an \
adjacent field. A predictable empty is better than a silent wrong.
3. Selection marks. ``☒`` selects the label spatially adjacent to it \
(same cell, same line, or the immediately adjacent non-empty line). \
``☐`` means unselected and never contributes a value. If a ``☒`` has \
no clear single adjacent label — e.g. it sits between two candidate \
labels with equal claim, or the candidate labels are separated from it \
by a heading or other content — treat it as unresolvable and emit \
``""``. Enum values must come from the allowed sets (``gender`` ∈ \
{_GENDER_LIST}; ``accidentLocation`` ∈ {_LOCATION_LIST}; \
``medicalInstitutionFields.healthFundMember`` ∈ {_FUND_LIST}) or \
``""``.
4. Dates (DDMMYYYY). Split into ``day`` / ``month`` / ``year`` (zero-\
padded). All three parts or none — if any part is missing or \
illegible, emit all three as ``""``.
5. Address. The row is either a street address (``street``, \
``houseNumber``, optional ``entrance``/``apartment``, ``city``, \
``postalCode``) or a PO Box (``poBox``). Fill the path that is present; \
leave the other path's fields as ``""``.
6. Section 5 is the medical-institution block. ``healthFundMember`` is \
the fund name whose checkbox is marked — ``כללית`` / ``מכבי`` / \
``מאוחדת`` / ``לאומית``. The separate "הנפגע חבר בקופת חולים" / \
"הנפגע אינו חבר בקופת חולים" row is a membership-status indicator and \
MUST NEVER populate ``healthFundMember``. ``natureOfAccident`` and \
``medicalDiagnoses`` are filled by the clinic after submission; on an \
applicant-submitted page they are ``""``.
"""


# A single, normally-filled form. Clean glyphs, clean dates, one marked
# checkbox per enum, a street address (PO-Box path left empty). No
# edge cases — the goal is to anchor the OCR→JSON mapping, not to
# demonstrate recovery from degraded OCR.
FEW_SHOT_OCR = """\
המוסד לביטוח לאומי

בקשה למתן טיפול רפואי לנפגע עבודה - עצמאי

תאריך קבלת הטופס בקופה
0 2 0 4 2 0 2 4
שנה חודש יום

תאריך מילוי הטופס
0 4 0 4 2 0 2 4
שנה חודש יום


# 1

תאריך הפגיעה

0 3 0 4 2 0 2 4

שנה חודש יום

2

פרטי התובע
שם משפחה
כהן

שם פרטי
דנה

ת.ז.
0 1 1 1 1 1 1 1 1

מין
☐ זכר ☒ נקבה

תאריך לידה
0 1 0 2 1 9 9 0

<table>
<tr><th>רחוב / תא דואר</th><th>מס׳ בית</th><th>כניסה</th><th>דירה</th><th>יישוב</th><th>מיקוד</th></tr>
<tr><td>הרצל</td><td>10</td><td></td><td></td><td>תל אביב</td><td>6100000</td></tr>
</table>

טלפון קווי

טלפון נייד
0 5 0 1 2 3 4 5 6 7

3
פרטי התאונה

סוג העבודה
מהנדסת תוכנה

בשעה
09:30

מקום התאונה:
☒ במפעל ☐ ת. דרכים בעבודה ☐ ת. דרכים בדרך לעבודה/מהעבודה ☐ תאונה בדרך ללא רכב ☐ אחר

כתובת מקום התאונה
דרך מנחם בגין 12, תל אביב

נסיבות הפגיעה / תאור התאונה
החלקתי על רצפה רטובה.

האיבר שנפגע
גב תחתון

4
הצהרה

שם המבקש
דנה כהן

5
למילוי ע״י המוסד הרפואי
☒ מאוחדת ☐ כללית ☐ מכבי ☐ לאומית
☐ הנפגע חבר בקופת חולים ☐ הנפגע אינו חבר בקופת חולים
"""

FEW_SHOT_JSON = {
    "lastName": "כהן",
    "firstName": "דנה",
    "idNumber": "011111111",
    "gender": "נקבה",
    "dateOfBirth": {"day": "01", "month": "02", "year": "1990"},
    "address": {
        "street": "הרצל",
        "houseNumber": "10",
        "entrance": "",
        "apartment": "",
        "city": "תל אביב",
        "postalCode": "6100000",
        "poBox": "",
    },
    "landlinePhone": "",
    "mobilePhone": "0501234567",
    "jobType": "מהנדסת תוכנה",
    "dateOfInjury": {"day": "03", "month": "04", "year": "2024"},
    "timeOfInjury": "09:30",
    "accidentLocation": "במפעל",
    "accidentAddress": "דרך מנחם בגין 12, תל אביב",
    "accidentDescription": "החלקתי על רצפה רטובה.",
    "injuredBodyPart": "גב תחתון",
    "signature": "",
    "formFillingDate": {"day": "04", "month": "04", "year": "2024"},
    "formReceiptDateAtClinic": {"day": "02", "month": "04", "year": "2024"},
    "medicalInstitutionFields": {
        "healthFundMember": "מאוחדת",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}


def build_messages(markdown: str) -> list[dict[str, str]]:
    """Return the full chat-completions messages list for one extraction call.

    Order: system → few-shot user (markdown) → few-shot assistant (JSON) →
    real user turn. Structural constraints (field names, enum values,
    empty-string defaults) are carried by ``response_format=json_schema``
    in strict mode; the prompt does not restate them.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_OCR},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_JSON, ensure_ascii=False)},
        {"role": "user", "content": markdown},
    ]
