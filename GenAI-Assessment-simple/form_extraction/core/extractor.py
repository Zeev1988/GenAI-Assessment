"""Extract Form 283 fields from OCR text using Azure OpenAI structured outputs.

The OCR stage (`form_extraction.core.ocr`) produces a single
``=== FORM 283 SPATIAL EXTRACTION ===`` document containing every field
value pre-computed from Azure Document Intelligence polygon coordinates
(word bounding boxes for text fields, selection-mark polygons for checkboxes).

The LLM's job is minimal: copy every value in the spatial header verbatim
into the JSON schema, split DDMMYYYY dates into day/month/year parts, and
map checkbox labels to their enum values.
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

The OCR input begins with "=== FORM 283 SPATIAL EXTRACTION ===" which contains \
ALL field values pre-computed from Azure Document Intelligence polygon \
coordinates.  These values are AUTHORITATIVE and CORRECT.  Your only job is to \
copy them verbatim into the JSON output.

Field mapping (spatial header → JSON)
--------------------------------------
Dates — split DDMMYYYY from the header line into day/month/year:
  • formReceiptDateAtClinic ← "formReceiptDateAtClinic:" line
  • formFillingDate         ← "formFillingDate:" line
  • dateOfInjury            ← "dateOfInjury:" line
  • dateOfBirth             ← "dateOfBirth:" line

Identifiers:
  • idNumber      ← "idNumber:" line (copy digit string verbatim)
  • mobilePhone   ← "mobilePhone:" line
  • landlinePhone ← "landlinePhone:" line

Checkboxes (from the selected-checkbox list):
  • gender           ← "זכר" or "נקבה"; else ""
  • accidentLocation ← "במפעל" / "ת. דרכים בעבודה" / \
    "ת. דרכים בדרך לעבודה/מהעבודה" / "תאונה בדרך ללא רכב" / \
    "מחוץ למפעל" / "אחר"; else ""
  • medicalInstitutionFields.healthFundMember ← \
    "healthFundMember (resolved):" line ("כללית"/"מכבי"/"מאוחדת"/"לאומית" or "")

Free-text fields (from "-- Free-text fields --" section of header):
  • lastName            ← "lastName:" line
  • firstName           ← "firstName:" line
  • address.street      ← "street:" line
  • address.houseNumber ← "houseNumber:" line
  • address.entrance    ← "entrance:" line
  • address.apartment   ← "apartment:" line
  • address.city        ← "city:" line
  • address.postalCode  ← "postalCode:" line
  • address.poBox       ← always "" (not on this form)
  • jobType             ← "jobType:" line
  • timeOfInjury        ← "timeOfInjury:" line
  • accidentAddress     ← "accidentAddress:" line
  • accidentDescription ← "accidentDescription:" line
  • injuredBodyPart     ← "injuredBodyPart:" line
  • signature           ← "applicantName:" line (the printed name in the \
    declaration section)
  • medicalInstitutionFields.natureOfAccident  ← always "" (filled by clinic)
  • medicalInstitutionFields.medicalDiagnoses  ← always "" (filled by clinic)

Rules
-----
1. Copy every spatial-header value VERBATIM into the JSON.
2. "(not found)" or "(none — …)" in the header → "" in JSON \
   (empty string for all date sub-parts too).
3. DDMMYYYY dates: day = chars 1-2, month = chars 3-4, year = chars 5-8.
4. idNumber: copy verbatim including leading zeros, even if digit count ≠ 9. \
   A non-standard length is a data-quality signal — do NOT trim or pad.
5. Preserve original language (Hebrew stays Hebrew, English stays English).
6. If a field is genuinely blank (empty string in header), output "".  \
   Never return null, "N/A", "-", or placeholder text.
7. Output JSON only. No prose, no markdown fences.
"""

# ---------------------------------------------------------------------------
# Few-shot examples
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

-- Selected checkboxes (from Azure DI selection marks + directional label match) --
  [SELECTED] at (5.90, 3.05)  →  label: נקבה
  [SELECTED] at (5.10, 6.10)  →  label: במפעל

healthFundMember (resolved): (none — no fund checkbox selected)

-- Free-text fields (extracted from fixed coordinate regions) --
lastName:            כהן
firstName:           דנה
street:              הרצל
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
applicantName:       דנה כהן
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

# Edge-case example: missing dates, blank address, no health-fund checkbox,
# membership-status mark that does NOT set healthFundMember.
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

-- Selected checkboxes (from Azure DI selection marks + directional label match) --
  [SELECTED] at (5.90, 3.05)  →  label: זכר
  [SELECTED] at (4.80, 7.90)  →  label: הנפגע חבר בקופת חולים

healthFundMember (resolved): (none — no fund checkbox selected)

-- Free-text fields (extracted from fixed coordinate regions) --
lastName:            לוי
firstName:           רועי
street:              (not found)
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
applicantName:
"""

FEW_SHOT_NEG_JSON = {
    "lastName": "לוי",
    "firstName": "רועי",
    "idNumber": "0222222222",
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
        # Negative example: missing dates, no fund resolved, blank address &
        # signature.  Demonstrates "(not found)" / "(none — …)" → "" in JSON,
        # and that a membership-status checkbox does not set healthFundMember.
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
