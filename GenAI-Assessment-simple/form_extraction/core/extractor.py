"""Extract Form 283 fields from OCR text using Azure OpenAI structured outputs.

One LLM call with the `json_schema` response format. If Pydantic still rejects
the payload (rare under strict mode), we issue exactly one corrective re-ask.
No second pass, no "targeted" fix-up — the brief asks for extraction, not a
self-healing loop.
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
You extract fields from a Bituach Leumi (ביטוח לאומי) Form 283 — Report of \
an Accident at Work. You receive the OCR output of a single form and must \
return a JSON object that matches the provided schema.

Rules:
1. Copy values verbatim from the OCR. Do not invent, translate, or paraphrase.
2. For any field you cannot confidently locate, return an empty string "". \
   Never return null, "N/A", "-", or placeholder text.
3. Date parts are digit strings: day "01"-"31", month "01"-"12", year 4 digits. \
   Leave parts empty if the date is unknown or partial.
4. Preserve the original language of each value — Hebrew stays Hebrew, \
   English stays English.
5. Phone numbers are digit-only strings; strip spaces, hyphens, and parentheses.
6. The form uses checkboxes for gender (זכר / נקבה), health fund \
   (כללית / מכבי / מאוחדת / לאומית), and accident location \
   (במפעל / מחוץ למפעל / בדרך לעבודה / בדרך מהעבודה / ת. דרכים בעבודה / אחר). \
   Copy the label of the checked box verbatim. Because the OCR processes \
   right-to-left text, checkboxes on the same row may appear in reversed order; \
   identify the checked option by which label sits directly next to the \
   selection marker, not by its position in the list.
7. "jobType" is the free-text "סוג העבודה" occupation, NOT a checkbox.
8. "healthFundMember": use the value from the NORMALIZED_HEALTH_FUND line at \
   the top of the OCR — copy it verbatim. If that line is empty \
   (NORMALIZED_HEALTH_FUND: ), set "healthFundMember" to "". Do NOT derive \
   this field from the body text, the SELECTED CHECKBOXES list, or any \
   membership-status label ("הנפגע חבר/אינו חבר בקופת חולים" is NOT a fund name).
9. Only extract a value if the OCR places it inside or immediately adjacent \
   to that field's label, in the correct section of the form. Never pull a \
   value from one section to fill another — if the matching content isn't \
   there, the field stays "". \
   EXCEPTION: the section-3 sentence (rule 12) — values appear before labels.
10. If the OCR starts with a "SELECTED CHECKBOXES" block, treat it as the \
    sole authoritative source for gender and accidentLocation. \
    Do NOT use ☒/☐/:selected: symbols anywhere in the rest of the text — they \
    may be spatially displaced due to right-to-left processing. Match each label \
    in the SELECTED CHECKBOXES list to the appropriate schema field. \
    "healthFundMember" is handled separately by rule 8 (NORMALIZED_HEALTH_FUND).
11. The form uses individual digit boxes. Azure DI outputs them with spaces \
    and pipe/bracket separators between digits, e.g. "[1 ] 4 0 4 1 | 9 9 9" \
    or "[2| 0 0 5 1 9 9 9" or "0 3 0 3 1 9 7 4". \
    For any date field ("תאריך הפגיעה", "תאריך מילוי הטופס", \
    "תאריך קבלת הטופס", "תאריך לידה") or below a "שנה חודש יום" caption: \
    strip ALL non-digit characters (spaces, pipes "|", brackets "[]") from \
    the adjacent token sequence, concatenate the remaining digits, then split \
    the 8-digit result as DDMMYYYY → day, month, year. \
    Examples: "[1 ] 4 0 4 1 | 9 9 9" → "14041999" → day="14" month="04" year="1999"; \
    "[2| 0 0 5 1 9 9 9" → "20051999" → day="20" month="05" year="1999"; \
    "0 3 0 3 1 9 7 4" → "03031974" → day="03" month="03" year="1974".
12. ID number (ת. ז.): if the OCR starts with a "NORMALIZED_ID:" line, use \
    that value verbatim as "idNumber" — the preprocessing has already handled \
    any digit reversal caused by the bidi algorithm. Do NOT re-derive the ID \
    from the raw digit sequence in the body text. Copy the value exactly as \
    given, even if it is 10 digits — do not trim or modify it. \
    If no NORMALIZED_ID line is present, extract the digit run that follows \
    "ת. ז." and strip all non-digit characters. \
13. Section 3 contains a fill-in-the-blank sentence whose inline values appear \
    BEFORE their labels in the OCR due to right-to-left reordering: \
    "…שארעה לי [time] בשעה [job] כאשר עבדתי ב … [date] בתאריך סוג העבודה". \
    Extract "timeOfInjury" from the value adjacent to "בשעה", and "jobType" \
    from the value adjacent to "כאשר עבדתי ב" or just before "סוג העבודה", \
    even though the labels appear after the values.
14. Output JSON only. No prose, no markdown fences.
"""

# One compact Hebrew few-shot example. Uses the 8-digit date format and the
# section-3 sentence pattern that appear in real scanned forms.
FEW_SHOT_OCR = """\
NORMALIZED_HEALTH_FUND:
NORMALIZED_ID: 011111111
שם משפחה: כהן   שם פרטי: דנה
ת. ז.
1 2 3 4 5 6 7 8 9
מין: [ ] זכר  [X] נקבה
תאריך לידה
[0 1 0 2 1 9 9 0
שנה חודש יום
רחוב: הרצל   מספר בית: 10   ישוב: תל אביב   מיקוד: 6100000
טלפון נייד: 050-1234567
תאריך הפגיעה
[0 3 0 4 2 0 2 4
שנה חודש יום
אני מבקש לקבל עזרה רפואית בגין פגיעה בעבודה שארעה לי
09:30 כאשר עבדתי ב מהנדסת תוכנה

בתאריך

סוג העבודה
מקום התאונה: [X] במפעל  [ ] בדרך לעבודה
תיאור התאונה: החלקתי על רצפה רטובה
האיבר שנפגע: גב תחתון
חתימה: דנה כהן
תאריך מילוי הטופס
[0| 4 0 4 2 0 2 4
שנה חודש יום
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
    "accidentAddress": "",
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

# Negative few-shot.
# Key lessons demonstrated:
# (1) SELECTED CHECKBOXES lists "הנפגע חבר בקופת חולים" but NO fund name →
#     healthFundMember MUST be "" even though כללית/מאוחדת/מכבי/לאומית appear
#     in the body text as printed form labels.
# (2) ID has Hebrew prefix "עי" → Case A reversal gives 033452156.
FEW_SHOT_NEG_OCR = """\
NORMALIZED_HEALTH_FUND:
NORMALIZED_ID: 0222222222
SELECTED CHECKBOXES (authoritative — derived from spatial bounding-box data,
not from the ☒/☐ symbols below which may be misplaced due to RTL processing):
  • זכר
  • הנפגע חבר בקופת חולים

For every checkbox field use ONLY the labels listed above.
Ignore ☒/☐/:selected: markers in the OCR text that follows.
---
טופס 283 - הודעה על תאונת עבודה
קופת חולים (סמן את האפשרות המתאימה):
[ ] כללית   [ ] מכבי   [ ] מאוחדת   [ ] לאומית
שם משפחה: לוי   שם פרטי: רועי
ת. ז.
עי 7 6 5 1| 2 5 | 4 3 3 | 0
מין: [X] זכר  [ ] נקבה
תאריך לידה: 15 / 07 / 1985
טלפון נייד: 0529876543
סוג העבודה: חשמלאי
תאריך הפגיעה: 10 / 03 / 2024   שעת הפגיעה: 14:20
תיאור התאונה: נפלתי מסולם בזמן עבודה
האיבר שנפגע: כתף ימין
למילוי ע"י המוסד הרפואי:
  [X] הנפגע חבר בקופת חולים   [ ] כללית   [ ] מאוחדת   [ ] מכבי   [ ] לאומית
  [ ] הנפגע אינו חבר בקופת חולים
  מהות התאונה: ________________
  אבחנות רפואיות: ________________
  (חתימת הרופא וחותמת המוסד:)
"""

FEW_SHOT_NEG_JSON = {
    "lastName": "לוי",
    "firstName": "רועי",
    "idNumber": "0222222222",  # NORMALIZED_ID value copied verbatim (10 digits = validator will flag)
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
        # SELECTED CHECKBOXES lists "הנפגע חבר בקופת חולים" (membership status)
        # but NO fund name. healthFundMember is "" — do NOT infer fund from body.
        # Clinic section has blank lines → natureOfAccident and medicalDiagnoses "".
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
        # Negative example: partial form, blank clinic section. Demonstrates
        # that "" is the correct answer when a value isn't in its own section,
        # even if similar text appears elsewhere on the page.
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
