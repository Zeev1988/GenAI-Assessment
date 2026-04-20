"""Prompt templates and few-shot examples for Form 283 extraction.

All the text the LLM actually sees lives here so prompt tuning is a
one-file change and the extractor module stays focused on orchestration
(retry, validation, and the Azure OpenAI call).

Contents
--------
* Enum-list strings derived from the schema so the prompt can never
  disagree with what ``response_format=json_schema`` will accept.
* ``SYSTEM_PROMPT`` — the extraction instructions.
* ``FEW_SHOT_OCR`` / ``FEW_SHOT_JSON`` — a filled-form positive example.
* ``FEW_SHOT_NEG_OCR`` / ``FEW_SHOT_NEG_JSON`` — a sparse / edge-case
  example (missing dates, PO Box instead of street, membership-status
  checkbox that must NOT populate ``healthFundMember``).
* :func:`build_messages` — assembles the system + few-shot + user
  turns into the list passed to the chat-completions API.

Invariant: ``prompts.py`` imports from ``schemas.py``; the reverse
must never happen.
"""

from __future__ import annotations

import json

from form_extraction.core.schemas import (
    ACCIDENT_LOCATION_LABELS,
    GENDER_LABELS,
    HEALTH_FUND_LABELS,
)

# ---------------------------------------------------------------------------
# Enum lists — derived from the schema so the two can never drift.
# ---------------------------------------------------------------------------

_GENDER_LIST = " / ".join(f'"{label}"' for label in GENDER_LABELS)
_FUND_LIST = " / ".join(f'"{label}"' for label in HEALTH_FUND_LABELS)
_ACCIDENT_LOCATION_LIST = " / ".join(f'"{label}"' for label in ACCIDENT_LOCATION_LABELS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You extract fields from a Bituach Leumi (ביטוח לאומי) Form 283 — Request for \
Medical Treatment for a Work Injury — Self-Employed.

The OCR input begins with "=== FORM 283 SPATIAL EXTRACTION ===" and contains \
every field value pre-computed from Azure Document Intelligence polygon \
coordinates. Those values are AUTHORITATIVE. Copy them verbatim into the JSON.

Field mapping (spatial header line → JSON path)
-----------------------------------------------
Dates (split DDMMYYYY into {{day, month, year}}):
  formReceiptDateAtClinic  →  formReceiptDateAtClinic
  formFillingDate          →  formFillingDate
  dateOfInjury             →  dateOfInjury
  dateOfBirth              →  dateOfBirth

Identifiers & phones (copy digit string verbatim):
  idNumber        →  idNumber
  mobilePhone     →  mobilePhone
  landlinePhone   →  landlinePhone

Free-text fields (copy verbatim):
  lastName             →  lastName
  firstName            →  firstName
  street               →  address.street
  poBox                →  address.poBox
  houseNumber          →  address.houseNumber
  entrance             →  address.entrance
  apartment            →  address.apartment
  city                 →  address.city
  postalCode           →  address.postalCode
  jobType              →  jobType
  timeOfInjury         →  timeOfInjury
  accidentAddress      →  accidentAddress
  accidentDescription  →  accidentDescription
  injuredBodyPart      →  injuredBodyPart

Resolved checkbox lines (one pre-resolved value per enum field):
  "gender (resolved):"            →  gender (one of {_GENDER_LIST}, else "")
  "accidentLocation (resolved):"  →  accidentLocation \
(one of {_ACCIDENT_LOCATION_LIST}, else "")
  "healthFundMember (resolved):"  →  medicalInstitutionFields.healthFundMember \
(one of {_FUND_LIST}, else "")

Rules
-----
1. Copy every spatial-header value verbatim into its mapped JSON path.
2. "(not found)" or "(none — …)" in the header → "" in the JSON (every date \
   sub-part is also "").
3. DDMMYYYY dates: day = chars 1-2, month = chars 3-4, year = chars 5-8.
4. idNumber: copy verbatim including leading zeros, even if the digit count \
   is not 9 — a non-standard length is a data-quality signal the validator \
   will flag; do not trim or pad.
5. Schema fields with no corresponding header line → "". This covers \
   signature, medicalInstitutionFields.natureOfAccident, and \
   medicalInstitutionFields.medicalDiagnoses. Never invent values for them \
   from adjacent sections.
6. Output JSON only — no prose, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Few-shot examples
#
# Both examples are constructed to match the exact text that
# ocr._render_spatial_header emits. Keep them in sync with that function
# whenever its wording changes.
# ---------------------------------------------------------------------------

FEW_SHOT_OCR = """\
=== FORM 283 SPATIAL EXTRACTION ===

These values are pre-computed from Azure DI word coordinates.
Each field was extracted by collecting all words whose centre
falls inside a calibrated bounding-box region for that field.
Checkboxes are resolved from polygon-based selection marks.
They are authoritative — use them verbatim and do NOT
re-derive from the form body.

-- Dates (DDMMYYYY, extracted from fixed coordinate regions) --
formReceiptDateAtClinic: (not found)
formFillingDate:         04042024  (day=04  month=04  year=2024)
dateOfInjury:            03042024  (day=03  month=04  year=2024)
dateOfBirth:             01021990  (day=01  month=02  year=1990)

-- Identifiers & phones --
idNumber:                011111111
mobilePhone:             0501234567
landlinePhone:           (not found)

-- Selected checkboxes (from Azure DI selection marks + coordinate region lookup) --
  [SELECTED] at (5.90, 3.05)  →  label: נקבה
  [SELECTED] at (5.10, 6.10)  →  label: במפעל
  [SELECTED] at (5.67, 9.89)  →  label: מכבי

gender (resolved):           נקבה
accidentLocation (resolved): במפעל
healthFundMember (resolved): מכבי

-- Free-text fields (extracted from fixed coordinate regions) --
lastName:            כהן
firstName:           דנה
street:              הרצל
poBox:
houseNumber:         10
entrance:
apartment:
city:                תל אביב
postalCode:          6100000
jobType:             מהנדסת תוכנה
timeOfInjury:        09:30
accidentAddress:     דרך מנחם בגין 12, תל אביב
accidentDescription: החלקתי על רצפה רטובה
injuredBodyPart:     גב תחתון
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
    "accidentDescription": "החלקתי על רצפה רטובה",
    "injuredBodyPart": "גב תחתון",
    "signature": "",
    "formFillingDate": {"day": "04", "month": "04", "year": "2024"},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        "healthFundMember": "מכבי",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}

# Edge-case example: missing dates, PO Box instead of street name, and only
# the membership-status checkbox is marked in Section 5 (must NOT populate
# healthFundMember).
FEW_SHOT_NEG_OCR = """\
=== FORM 283 SPATIAL EXTRACTION ===

These values are pre-computed from Azure DI word coordinates.
Each field was extracted by collecting all words whose centre
falls inside a calibrated bounding-box region for that field.
Checkboxes are resolved from polygon-based selection marks.
They are authoritative — use them verbatim and do NOT
re-derive from the form body.

-- Dates (DDMMYYYY, extracted from fixed coordinate regions) --
formReceiptDateAtClinic: (not found)
formFillingDate:         (not found)
dateOfInjury:            10032024  (day=10  month=03  year=2024)
dateOfBirth:             15071985  (day=15  month=07  year=1985)

-- Identifiers & phones --
idNumber:                0222222222
mobilePhone:             0529876543
landlinePhone:           (not found)

-- Selected checkboxes (from Azure DI selection marks + coordinate region lookup) --
  [SELECTED] at (5.90, 3.05)  →  label: זכר
  [SELECTED] at (4.80, 7.90)  →  label: הנפגע חבר בקופת חולים

gender (resolved):           זכר
accidentLocation (resolved): (none — no accident-location checkbox selected)
healthFundMember (resolved): (none — no fund checkbox selected)

-- Free-text fields (extracted from fixed coordinate regions) --
lastName:            לוי
firstName:           רועי
street:              (not found)
poBox:               5678
houseNumber:         (not found)
entrance:
apartment:
city:                (not found)
postalCode:
jobType:             חשמלאי
timeOfInjury:        14:20
accidentAddress:
accidentDescription: נפלתי מסולם בזמן עבודה
injuredBodyPart:     כתף ימין
"""

FEW_SHOT_NEG_JSON = {
    "lastName": "לוי",
    "firstName": "רועי",
    "idNumber": "0222222222",
    "gender": "זכר",
    "dateOfBirth": {"day": "15", "month": "07", "year": "1985"},
    "address": {
        "street": "", "houseNumber": "", "entrance": "", "apartment": "",
        "city": "", "postalCode": "", "poBox": "5678",
    },
    "landlinePhone": "",
    "mobilePhone": "0529876543",
    "jobType": "חשמלאי",
    "dateOfInjury": {"day": "10", "month": "03", "year": "2024"},
    "timeOfInjury": "14:20",
    "accidentLocation": "",
    "accidentAddress": "",
    "accidentDescription": "נפלתי מסולם בזמן עבודה",
    "injuredBodyPart": "כתף ימין",
    "signature": "",
    "formFillingDate": {"day": "", "month": "", "year": ""},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------


def build_messages(ocr_text: str) -> list[dict[str, str]]:
    """Return the full chat-completions messages list for an extraction call.

    Order:
      1. system prompt
      2. positive few-shot example (filled form, fund resolved)
      3. negative / edge-case few-shot (missing dates, PO Box, unresolved fund)
      4. the real OCR payload as the final user turn
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Positive example: filled form, fund resolved, straightforward copy.
        {"role": "user", "content": f"OCR content:\n{FEW_SHOT_OCR}"},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_JSON, ensure_ascii=False)},
        # Negative example: missing dates, blank address, membership-status
        # checkbox that must NOT populate healthFundMember.
        {"role": "user", "content": f"OCR content:\n{FEW_SHOT_NEG_OCR}"},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_NEG_JSON, ensure_ascii=False)},
        # Real request.
        {"role": "user", "content": f"OCR content:\n{ocr_text}"},
    ]
