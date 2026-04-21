"""System prompt and few-shot for Form 283 extraction."""

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
of a scanned form. Selection marks appear as ``☒`` (checked) / ``☐`` \
(empty). Date boxes render as spaced digit runs, e.g. ``0 3 0 4 2 0 2 4``.

Output
------
A single JSON object matching the provided schema. Use ``""`` for any \
value that is not clearly filled. Dates split into ``{{"day": "DD", \
"month": "MM", "year": "YYYY"}}``.

Rules
-----
1. Copy verbatim. Preserve Hebrew exactly; do not translate or reformat. \
Digit fields have no spaces or separators.
2. Empty stays empty. If a field is blank or illegible, emit ``""``. Do \
not guess or copy from adjacent fields.
3. Selection marks. ``☒`` selects the label adjacent to it. Enum values \
must come from the allowed sets (``gender`` ∈ {_GENDER_LIST}; \
``accidentLocation`` ∈ {_LOCATION_LIST}; \
``medicalInstitutionFields.healthFundMember`` ∈ {_FUND_LIST}) or ``""``.
4. Dates (DDMMYYYY). Split into ``day``/``month``/``year`` (zero-padded). \
All three parts or all three ``""``.
5. Address. Either a street address (``street``, ``houseNumber``, \
optional ``entrance``/``apartment``, ``city``, ``postalCode``) or a PO \
Box (``poBox``). Leave the unused path empty.
6. Section 5 is the medical-institution block. ``healthFundMember`` is \
the fund name whose checkbox is marked. The separate \
"הנפגע חבר/אינו חבר" row is a membership indicator and MUST NEVER \
populate ``healthFundMember``. ``natureOfAccident`` and \
``medicalDiagnoses`` are filled by the clinic; on an applicant-submitted \
page they are ``""``.
"""


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
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_OCR},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_JSON, ensure_ascii=False)},
        {"role": "user", "content": markdown},
    ]
