# Form 283 Extractor

Lean Part 1 solution for the GenAI take-home: upload a Bituach Leumi Form 283
(PDF/JPG/PNG), run Azure Document Intelligence for OCR, feed the OCR output to
GPT-4o with a JSON-schema structured output, and return the fields as JSON.

## Data flow

```
                 +--------------------+     spatial         +------------------+
 PDF / JPG / PNG | Azure Document     |     header          | GPT-4o           |
 --------------> |  Intelligence      | -----------------> |  (json_schema,    |
                 |  prebuilt-layout   |   word polygons +   |   strict mode)   |
                 |  + coordinate      |   selection marks   |                  |
                 |  region lookup     |                     |                  |
                 +--------------------+                     +---------+--------+
                                                                      |
                                                                      v
                                                            +------------------+
                                                            | Pydantic         |
                                                            |   ExtractedForm  |
                                                            +---------+--------+
                                                                      |
                                                                      v
                                                            +------------------+
                                                            | validate()       |
                                                            |   formats +      |
                                                            |   grounding      |
                                                            +---------+--------+
                                                                      |
                                                                      v
                                                            +------------------+
                                                            | Streamlit UI     |
                                                            |   JSON · OCR · ⚠ |
                                                            +------------------+
```

## Layout

```
form_extraction/
  core/              # framework-agnostic library
    config.py        # env-loaded settings (pydantic-settings)
    schemas.py       # Pydantic models + Hebrew label map + OpenAI json_schema;
                     #   single source of truth for enum labels
    field_regions.py # calibrated bounding-box regions for every field and
                     #   checkbox on page 1 of Form 283
    ocr.py           # Azure DI (prebuilt-layout) wrapper + coordinate-based
                     #   field + checkbox extraction
    extractor.py     # one GPT-4o call + positive/negative few-shots, one
                     #   retry on Pydantic error
    validate.py      # format checks + grounding check against OCR text
    pipeline.py      # OCR -> extract -> validate
  ui/                # Streamlit entrypoint; imports only from core
    app.py
tests/
  test_schema.py     # round-trip + json_schema strictness
  test_validate.py   # format rules, completeness, and grounding
  test_e2e.py        # parametrised E2E against the 3 known sample PDFs
                     #   (gated behind RUN_AZURE_TESTS=1; OCR results cached
                     #    under tests/fixtures/ocr_cache/)
  test_live.py       # gated smoke test against a real sample PDF
```

The public library surface is re-exported from `form_extraction/__init__.py`,
so `from form_extraction import run, ExtractedForm` works. Internal callers
should import from `form_extraction.core.*`.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
# fill in AZURE_* values from the assignment email
```

## Run

```bash
streamlit run form_extraction/ui/app.py
```

Upload a PDF/JPG/PNG (or click the "Load sample" button if you have the
provided `phase1_data/` samples next to the repo). The UI shows the
extracted JSON, the raw OCR text, and validation issues in three tabs.
Toggle "Show Hebrew field labels" to re-label the JSON with the Hebrew
keys from the assignment spec; both language variants are downloadable.

## Test

```bash
pytest                               # unit tests (no Azure calls)
pytest -m "not integration"          # explicitly skip the gated E2E tests
RUN_AZURE_TESTS=1 pytest             # include the live E2E tests
```

The E2E suite (`tests/test_e2e.py`) caches Azure DI OCR output under
`tests/fixtures/ocr_cache/<stem>.txt`.  First run requires
`RUN_AZURE_TESTS=1` and live Azure DI + Azure OpenAI credentials; subsequent
runs skip the Document Intelligence call and only hit Azure OpenAI.
Sample PDFs are looked up in `tests/test_data/phase1_data/` and, as a
fallback, in `phase1_data/` at the repo root.

## Example output

Given a filled form, the extractor returns JSON matching the spec exactly:

```json
{
  "lastName": "כהן",
  "firstName": "דנה",
  "idNumber": "123456789",
  "gender": "נקבה",
  "dateOfBirth": {"day": "01", "month": "02", "year": "1990"},
  "address": {
    "street": "הרצל", "houseNumber": "10", "entrance": "", "apartment": "",
    "city": "תל אביב", "postalCode": "6100000", "poBox": ""
  },
  "landlinePhone": "",
  "mobilePhone": "0501234567",
  "jobType": "מהנדסת תוכנה",
  "dateOfInjury": {"day": "03", "month": "04", "year": "2024"},
  "timeOfInjury": "09:30",
  "accidentLocation": "במפעל",
  "...": "..."
}
```

## Design notes

**Coordinate-based OCR, not prompt-based field location.** Form 283 is a
fixed-layout government form, so every field has a stable bounding box on
the page.  `ocr.py` collects all words whose centre falls inside a
pre-calibrated region (`field_regions.PAGE_1`) and sorts them RTL into
reading order.  Checkboxes are resolved the same way: each selection mark
polygon is looked up against `field_regions.CHECKBOXES_PAGE_1`.  The LLM
never sees the raw Markdown OCR stream — it only sees a pre-rendered
`=== FORM 283 SPATIAL EXTRACTION ===` header with field-name → value
pairs, so field location is fully deterministic.

**One LLM call per form.** The extractor issues a single GPT-4o call with
`response_format=json_schema` (strict mode). The model's only job is to
copy spatial-header values into typed JSON and split DDMMYYYY dates. On the
rare case Pydantic rejects the payload we retry once with the validation
error appended to the conversation. No targeted re-ask loop on
business-rule failures — that's cost and complexity the brief doesn't ask
for.

**Single source of truth for enum labels.** `schemas.py` owns the
`GENDER_LABELS`, `HEALTH_FUND_LABELS`, and `ACCIDENT_LOCATION_LABELS`
tuples; the prompt enum lists are built from them at import time and
`field_regions.py` asserts they match.  The prompt, the strict
`json_schema` response format, and the coordinate-to-label map can never
drift out of sync.

**Positive + negative few-shot.** The prompt includes two worked examples:
one filled form with a fund checkbox resolved, and one edge case with
missing dates, a blank address, and only the "is a member" membership-status
checkbox marked (which must NOT populate `healthFundMember`). The negative
example anchors the "leave it empty" behaviour that `strict` mode otherwise
bullies the model out of.

**Two layers of validation.** `validate.py` runs format rules (ID length,
phone shape, HH:MM time, real calendar dates) and a **grounding check** that
flags free-text values which don't appear as substrings of the OCR — the
cheap, deterministic guardrail against hallucinations.  Enum fields are
not grounded (they come from checkbox resolution, not the OCR body).
Neither layer rewrites values; the UI surfaces issues and the user decides.

**Accuracy check is human-in-the-loop.** The UI's "OCR text" tab shows the
spatial-extraction header next to the extracted JSON so the reviewer can
eyeball correctness — still the most reliable accuracy signal, and the thing
the assignment explicitly asks for.

**Missing fields are empty strings** per the spec. The Pydantic models
default every field to `""`, and the LLM is instructed to leave them that
way when a field isn't visible.

**Hebrew and English** are both handled by keeping the OCR content in its
native script and instructing the LLM never to translate values. The UI
toggle relabels the keys of the JSON output into Hebrew without touching
the values.

**PII-safe structured logging** via stdlib `logging` (no external deps).
The UI sets up `basicConfig` at INFO; each pipeline stage logs a
`name.event key=value` line with byte counts, char counts, elapsed ms, and
field counts.  Raw field values are logged only at DEBUG level.

## Known limitations

- Bounding-box regions are hard-coded to the provided Form 283 template.
  A revised form with shifted fields would need `field_regions.py`
  re-calibrated.
- The grounding check is substring-based; a hallucination that happens to
  collide with text elsewhere on the form (e.g. a first name pulled into a
  diagnosis field) won't be flagged. The prompt's "copy verbatim from the
  spatial header" rule is the primary defence; grounding is the backstop.
- Strict `json_schema` mode forces every property to be present. We rely on
  the prompt + few-shots to produce `""` for absent values.
- Azure Document Intelligence accuracy degrades on scans rotated >15°. The
  prebuilt-layout model is the best we get without a custom model.

## Not included on purpose

- No persistent cache, correlation IDs, PII redactor, retry wrapper, or
  LLM-as-judge. The brief is a prototype, not a production microservice.
- No synthetic-PDF generator. The three provided samples are the
  authoritative test fixtures.
- No sanitiser layer that "fixes" OCR misreads. Rewriting `6512345678` to
  `0512345678` is invention, and the prompt explicitly forbids it.
- No signature image capture. The schema keeps a `signature` key for
  parity with the assignment spec but it is always `""` — the applicant's
  name lives in `firstName`/`lastName`.
