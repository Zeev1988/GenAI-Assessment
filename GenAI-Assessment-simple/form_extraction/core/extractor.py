"""Extract Form 283 fields from OCR text using Azure OpenAI structured outputs.

The OCR stage (`form_extraction.core.ocr`) produces a two-part document:

  1. ``=== FORM 283 SPATIAL EXTRACTION ===`` — authoritative key/value pairs
     for every dated / numbered / checkbox field, derived from Azure Document
     Intelligence's Markdown stream (label-anchored regex for dates / ID /
     phones) and its polygon-based selection marks (for checkboxes).  Each
     line is shaped like:

         formReceiptDateAtClinic: 02021999  (day=02  month=02  year=1999)   [label 'תאריך קבלת הטופס בקופה']

     and the checkbox section lists one ``[SELECTED]`` line per selected mark
     with the resolved label and a single ``healthFundMember (resolved): …``
     line at the bottom.

  2. ``=== FORM BODY (markdown OCR) ===`` — the raw Markdown OCR, useful only
     for free-text fields (names, address parts, job type, accident
     description, injured body part, signature, clinic free-text).

The LLM's job is therefore very narrow: copy the spatial-header values
verbatim into the schema, and read the Markdown body only for the
descriptive free-text fields.  No RTL reasoning, no date parsing, no
checkbox inference.
"""

from __future__ import annotations

import json
import logging
import time

from openai import AzureOpenAI
from pydantic import ValidationError

from form_extraction.core.config import Settings, get_settings
from form_extraction.core.schemas import ExtractedForm, openai_json_schema

log = logging.getLogger("form_extraction.extractor")

SYSTEM_PROMPT = """\
You extract fields from a Bituach Leumi (ביטוח לאומי) Form 283 — Request for \
Medical Treatment for a Work Injury — Self-Employed.

The OCR input has two parts:

A. "=== FORM 283 SPATIAL EXTRACTION ===" — authoritative key/value pairs \
derived from the page's polygon coordinates.  These are PRE-COMPUTED and \
CORRECT.  For every field listed in this header, COPY THE VALUE VERBATIM \
into the JSON output.  Never re-derive these values from the Markdown body.

B. "=== FORM BODY (markdown OCR) ===" — the raw Markdown OCR. Use this \
ONLY for free-text / descriptive fields that are NOT listed in the spatial \
header (names, address parts, job type, accident address, accident \
description, injured body part, signature, natureOfAccident, \
medicalDiagnoses).

Field source map
----------------
From the spatial header (copy exactly):
  • formReceiptDateAtClinic ← "formReceiptDateAtClinic:" line (DDMMYYYY → \
    day/month/year parts).
  • formFillingDate         ← "formFillingDate:" line.
  • dateOfInjury            ← "dateOfInjury:" line.
  • dateOfBirth             ← "dateOfBirth:" line.
  • idNumber                ← "idNumber:" line (string of digits, as given).
  • mobilePhone             ← "mobilePhone:" line.
  • landlinePhone           ← "landlinePhone:" line.
  • gender                  ← look in the selected-checkbox list for \
    "זכר" or "נקבה"; else "".
  • accidentLocation        ← look in the selected-checkbox list for one of \
    "במפעל" / "מחוץ למפעל" / "בדרך לעבודה" / "בדרך מהעבודה" / \
    "ת. דרכים בעבודה" / "אחר"; else "".
  • medicalInstitutionFields.healthFundMember ← the \
    "healthFundMember (resolved):" line ("כללית" / "מכבי" / "מאוחדת" / \
    "לאומית" or "" if none).

From the Markdown body:
  • lastName, firstName — from SECTION 2 ("פרטי התובע").
  • address.{street, houseNumber, entrance, apartment, city, postalCode, \
    poBox} — from the address table in SECTION 2.
  • jobType — from SECTION 3, before "סוג העבודה" (or after "כאשר עבדתי ב").
  • timeOfInjury — from SECTION 3, the token before "בשעה" (HH:MM format).
  • accidentAddress — from SECTION 3, the text after "כתובת מקום התאונה".
  • accidentDescription — from SECTION 3, the free-text sentence for \
    "נסיבות הפגיעה / תאור התאונה".
  • injuredBodyPart — from SECTION 3, after "האיבר שנפגע".
  • signature — from SECTION 4, the handwritten name line (appears above or \
    below "שם המבקש" / "חתימה"). If the page shows no handwritten name, "".
  • medicalInstitutionFields.natureOfAccident — from SECTION 5, text after \
    "מהות התאונה (אבחנות רפואיות):".
  • medicalInstitutionFields.medicalDiagnoses — from SECTION 5, any \
    additional diagnostic free-text.  If nothing is written, "".

Rules
-----
1. Copy every value in the spatial header VERBATIM.  If the header shows \
   "(not found)" or "(none — …)", use "" for that field (all its sub-parts \
   for dates).
2. Dates in the spatial header are always DDMMYYYY: day = chars 1-2, month \
   = chars 3-4, year = chars 5-8.  Split them exactly that way.
3. Never infer a checkbox value from ☐/☒/:selected: symbols in the Markdown \
   body — those are unreliable under RTL.  Use ONLY the selected-checkbox \
   list and the "healthFundMember (resolved)" line from the spatial header.
4. The spatial header is authoritative for idNumber.  Copy its digits \
   verbatim, INCLUDING any leading zeros and EVEN IF the digit count is \
   not 9.  A 10-digit ID (or any non-standard length) is a legitimate \
   data-quality signal that a downstream validator will flag — do NOT \
   truncate, pad, re-order, or re-derive the value from the Markdown body.
5. For free-text fields that are genuinely absent from the Markdown body, \
   return "".  Never return null, "N/A", "-", or placeholder text.
6. Preserve the original language of every value — Hebrew stays Hebrew, \
   English stays English.
7. Output JSON only. No prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Few-shot examples built around the new spatial-header OCR format.
# ---------------------------------------------------------------------------

FEW_SHOT_OCR = """\
=== FORM 283 SPATIAL EXTRACTION ===

These values are pre-computed from the Azure DI text stream
(after spaced-digit collapse) and from polygon-based selection
marks.  They are authoritative for every date / ID / phone /
checkbox field on Form 283 — use them verbatim and do NOT
re-derive from the markdown body below.

-- Dates (DDMMYYYY, extracted from the markdown after each label) --
formReceiptDateAtClinic: (not found)   [label 'תאריך קבלת הטופס בקופה']
formFillingDate:         04042024  (day=04  month=04  year=2024)   [label 'תאריך מילוי הטופס']
dateOfInjury:            03042024  (day=03  month=04  year=2024)   [label 'תאריך הפגיעה']
dateOfBirth:             01021990  (day=01  month=02  year=1990)   [label 'תאריך לידה']

-- Identifiers & phones --
idNumber:                011111111   [label 'ת.ז.']
mobilePhone:             0501234567   [label 'טלפון נייד']
landlinePhone:           (not found)   [label 'טלפון קווי']

-- Selected checkboxes (from Azure DI selection marks + directional label match) --
  [SELECTED] at (5.90, 3.05)  →  label: נקבה
  [SELECTED] at (5.10, 6.10)  →  label: במפעל

healthFundMember (resolved): (none — no fund checkbox selected)

Rules for the extractor:
  • For every date / ID / phone / gender / accidentLocation /
    healthFundMember field above, copy the value from this header.
  • If the header says '(not found)' or '(none...)' the corresponding
    field must be "" (empty string).

=== FORM BODY (markdown OCR) ===

=== FORM HEADER (formReceiptDateAtClinic, formFillingDate) ===
המוסד לביטוח לאומי
תאריך מילוי הטופס
04042024
שנה חודש יום

=== SECTION 1: תאריך הפגיעה (dateOfInjury) ===
תאריך הפגיעה
03042024
שנה חודש יום

=== SECTION 2: פרטי התובע (lastName, firstName, idNumber, gender, dateOfBirth, address, phones) ===
שם משפחה
כהן
שם פרטי
דנה
ת. ז.
1 2 3 4 5 6 7 8 9
מין
☒ נקבה  ☐ זכר
01021990
שנה חודש יום

| רחוב | מספר בית | כניסה | דירה | יישוב | מיקוד |
| הרצל | 10 |  |  | תל אביב | 6100000 |

טלפון נייד
0501234567

=== SECTION 3: פרטי התאונה (timeOfInjury, jobType, accidentLocation, accidentAddress, accidentDescription, injuredBodyPart) ===
אני מבקש לקבל עזרה רפואית בגין פגיעה בעבודה שארעה לי
09:30 כאשר עבדתי ב מהנדסת תוכנה
סוג העבודה
מקום התאונה: ☒ במפעל
כתובת מקום התאונה
דרך מנחם בגין 12, תל אביב
נסיבות הפגיעה / תאור התאונה
החלקתי על רצפה רטובה
האיבר שנפגע
גב תחתון

=== SECTION 4: הצהרה (signature) ===
דנה כהן
שם המבקש
חתימה X

=== SECTION 5: למילוי ע"י המוסד הרפואי (healthFundMember, natureOfAccident, medicalDiagnoses) ===
☐ כללית  ☐ מכבי  ☐ מאוחדת  ☐ לאומית
מהות התאונה (אבחנות רפואיות):
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
    "signature": "דנה כהן",
    "formFillingDate": {"day": "04", "month": "04", "year": "2024"},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}

# Negative / edge-case few-shot.
# Key lessons:
# (1) The spatial header's "(not found)" and "(none — …)" mean "" in JSON.
# (2) The selected-checkbox list contains "הנפגע חבר בקופת חולים" (a
#     membership-status label, NOT a fund name) but no fund is resolved →
#     healthFundMember = "".
# (3) accidentLocation is absent from the selected list → "".
# (4) The Markdown body contains ☐ marks next to fund names — IGNORE them;
#     only the resolved line in the spatial header counts.
FEW_SHOT_NEG_OCR = """\
=== FORM 283 SPATIAL EXTRACTION ===

-- Dates (DDMMYYYY, extracted from the markdown after each label) --
formReceiptDateAtClinic: (not found)   [label 'תאריך קבלת הטופס בקופה']
formFillingDate:         (not found)   [label 'תאריך מילוי הטופס']
dateOfInjury:            10032024  (day=10  month=03  year=2024)   [label 'תאריך הפגיעה']
dateOfBirth:             15071985  (day=15  month=07  year=1985)   [label 'תאריך לידה']

-- Identifiers & phones --
idNumber:                0222222222   [label 'ת.ז.']
mobilePhone:             0529876543   [label 'טלפון נייד']
landlinePhone:           (not found)   [label 'טלפון קווי']

-- Selected checkboxes (from Azure DI selection marks + directional label match) --
  [SELECTED] at (5.90, 3.05)  →  label: זכר
  [SELECTED] at (4.80, 7.90)  →  label: הנפגע חבר בקופת חולים

healthFundMember (resolved): (none — no fund checkbox selected)

=== FORM BODY (markdown OCR) ===

=== FORM HEADER (formReceiptDateAtClinic, formFillingDate) ===
טופס 283 - בקשה למתן טיפול רפואי

=== SECTION 1: תאריך הפגיעה (dateOfInjury) ===
10032024
שנה חודש יום

=== SECTION 2: פרטי התובע (lastName, firstName, idNumber, gender, dateOfBirth, address, phones) ===
שם משפחה
לוי
שם פרטי
רועי
ת. ז.
0 2 2 2 2 2 2 2 2 2
מין
☒ זכר
15071985
שנה חודש יום

טלפון נייד
0529876543

=== SECTION 3: פרטי התאונה (timeOfInjury, jobType, accidentLocation, accidentAddress, accidentDescription, injuredBodyPart) ===
14:20 כאשר עבדתי ב חשמלאי
סוג העבודה
נסיבות הפגיעה / תאור התאונה
נפלתי מסולם בזמן עבודה
האיבר שנפגע
כתף ימין

=== SECTION 4: הצהרה (signature) ===
שם המבקש
חתימה X

=== SECTION 5: למילוי ע"י המוסד הרפואי (healthFundMember, natureOfAccident, medicalDiagnoses) ===
☒ הנפגע חבר בקופת חולים
☐ כללית  ☐ מאוחדת  ☐ מכבי  ☐ לאומית
מהות התאונה (אבחנות רפואיות):
"""

FEW_SHOT_NEG_JSON = {
    "lastName": "לוי",
    "firstName": "רועי",
    "idNumber": "0222222222",  # copied verbatim from spatial header
    "gender": "זכר",
    "dateOfBirth": {"day": "15", "month": "07", "year": "1985"},
    "address": {
        "street": "", "houseNumber": "", "entrance": "", "apartment": "",
        "city": "", "postalCode": "", "poBox": "",
    },
    "landlinePhone": "",
    "mobilePhone": "0529876543",
    "jobType": "חשמלאי",
    "dateOfInjury": {"day": "10", "month": "03", "year": "2024"},
    "timeOfInjury": "14:20",
    "accidentLocation": "",  # not in the selected-checkbox list
    "accidentAddress": "",
    "accidentDescription": "נפלתי מסולם בזמן עבודה",
    "injuredBodyPart": "כתף ימין",
    "signature": "",  # no handwritten name present
    "formFillingDate": {"day": "", "month": "", "year": ""},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        # Spatial header says "(none — no fund checkbox selected)" → ""
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": "",
    },
}


def extract(ocr_text: str, settings: Settings | None = None) -> ExtractedForm:
    """Return an ExtractedForm built from one (or at most two) LLM calls."""
    s = settings or get_settings()
    if not s.azure_openai_endpoint or not s.azure_openai_key.get_secret_value():
        raise RuntimeError(
            "Azure OpenAI is not configured. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY."
        )

    schema = openai_json_schema()
    client = AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_key.get_secret_value(),
        api_version=s.azure_openai_api_version,
        timeout=s.request_timeout_s,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Positive example: filled form, straightforward extraction.
        {"role": "user", "content": f"OCR content:\n----- BEGIN -----\n{FEW_SHOT_OCR}\n----- END -----"},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_JSON, ensure_ascii=False)},
        # Negative example: missing dates in header, no fund resolved, blank
        # signature & clinic section.  Demonstrates that "(not found)" /
        # "(none — …)" in the spatial header → "" in JSON, and that body-text
        # ☐/☒ symbols never override the header.
        {"role": "user", "content": f"OCR content:\n----- BEGIN -----\n{FEW_SHOT_NEG_OCR}\n----- END -----"},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_NEG_JSON, ensure_ascii=False)},
        # Real request.
        {"role": "user", "content": f"OCR content:\n----- BEGIN -----\n{ocr_text}\n----- END -----"},
    ]

    log.info("extract.start ocr_chars=%d", len(ocr_text))
    t0 = time.perf_counter()
    payload = _call(client, s.azure_openai_deployment_extract, messages, schema)
    log.info("extract.llm_response %s", json.dumps(payload, ensure_ascii=False))
    try:
        form = ExtractedForm.model_validate(payload)
        log.info("extract.done elapsed_ms=%d", int((time.perf_counter() - t0) * 1000))
        return form
    except ValidationError as exc:
        # One corrective re-ask on the off-chance strict mode still produces
        # a payload Pydantic rejects (extra field trimmed post-JSON, etc.).
        log.warning("extract.retry reason=%s", exc.error_count())
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous JSON failed schema validation:\n"
                    f"{exc}\n"
                    "Return a corrected JSON that matches the schema exactly. "
                    "Keep every field you had right; only fix the validation errors. "
                    "Output JSON only."
                ),
            }
        )
        payload = _call(client, s.azure_openai_deployment_extract, messages, schema)
        form = ExtractedForm.model_validate(payload)
        log.info("extract.done retried=1 elapsed_ms=%d", int((time.perf_counter() - t0) * 1000))
        return form


def _call(
    client: AzureOpenAI,
    deployment: str,
    messages: list[dict[str, str]],
    schema: dict,
) -> dict:
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ExtractedForm", "schema": schema, "strict": True},
        },
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Azure OpenAI returned an empty response.")
    return json.loads(content)
