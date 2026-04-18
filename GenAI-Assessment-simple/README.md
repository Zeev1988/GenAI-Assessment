# Form 283 Extractor

Lean Part 1 solution for the GenAI take-home: upload a Bituach Leumi Form 283
(PDF/JPG/PNG), run Azure Document Intelligence for OCR, feed the OCR output to
GPT-4o with a JSON-schema structured output, and return the fields as JSON.

## Data flow

```
                 +--------------------+     markdown        +------------------+
 PDF / JPG / PNG | Azure Document     | -----------------> | GPT-4o            |
 --------------> |  Intelligence      |   OCR text          |  (json_schema,   |
                 |  prebuilt-layout   |                     |   strict mode)   |
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
  core/           # framework-agnostic library
    config.py     # env-loaded settings (pydantic-settings)
    schemas.py    # Pydantic models + Hebrew label map + OpenAI json_schema
    ocr.py        # Azure Document Intelligence (prebuilt-layout) wrapper
    extractor.py  # one GPT-4o call + positive/negative few-shots, one retry on Pydantic error
    validate.py   # format checks + grounding check against OCR text
    pipeline.py   # OCR -> extract -> validate
  ui/             # Streamlit entrypoint; imports only from core
    app.py
tests/
  test_schema.py      # round-trip + json_schema strictness + public re-exports
  test_validate.py    # format rules, completeness, and grounding
  test_live.py        # gated E2E against a real sample PDF
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
RUN_AZURE_TESTS=1 pytest             # plus the live E2E test
pytest -m "not integration"          # explicitly skip the live test
```

Place `283_ex*.pdf` samples in `phase1_data/` (sibling of the repo root) so
`test_live.py` can find them. The gated test skips cleanly when they're
absent.

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

**One LLM call per form.** The extractor issues a single GPT-4o call with
`response_format=json_schema` (strict mode). On the rare case that Pydantic
rejects the payload we retry once. There is no targeted re-ask loop on
business-rule failures — that's cost and complexity the brief doesn't ask
for.

**Positive + negative few-shot.** The prompt includes two worked examples:
one filled form, and one with a blank clinic section. The negative example
anchors the "leave it empty" behavior that `strict` mode otherwise bullies
the model out of.

**Two layers of validation.** `validate.py` runs format rules (ID length,
phone shape, HH:MM time, real calendar dates) and a **grounding check** that
flags free-text values which don't appear as substrings of the OCR — the
cheap, deterministic guardrail against hallucinations. Neither layer rewrites
values; the UI surfaces issues and the user decides.

**Accuracy check is human-in-the-loop.** The UI's "OCR text" tab shows the
raw Markdown-formatted OCR next to the extracted JSON so the reviewer can
eyeball correctness — still the most reliable accuracy signal, and the thing
the assignment explicitly asks for.

**Missing fields are empty strings** per the spec. The Pydantic models
default every field to `""`, and the LLM is instructed to leave them that
way when a field isn't visible.

**Hebrew and English** are both handled by keeping the OCR content in its
native script and instructing the LLM never to translate values. The UI
toggle relabels the keys of the JSON output into Hebrew without touching
the values.

**Structured logging** via stdlib `logging` (no external deps). The UI sets
up `basicConfig` at INFO; each pipeline stage logs a `name.event key=value`
line with byte counts, char counts, and elapsed ms.

## Known limitations

- The grounding check is substring-based; a hallucination that happens to
  collide with text elsewhere on the form (e.g. a first name pulled into a
  diagnosis field) won't be flagged. The prompt's "do not pull values from
  other sections" rule is the primary defense; grounding is the backstop.
- Strict `json_schema` mode forces every property to be present. We rely on
  the prompt + few-shots to produce `""` for absent values.
- Azure Document Intelligence accuracy degrades on scans rotated >15°. The
  prebuilt-layout model is the best we get without a custom model.

## Not included on purpose

- No persistent cache, correlation IDs, PII redactor, retry wrapper, or
  LLM-as-judge. The brief is a prototype, not a production microservice.
  See `REMOVED_FOR_PRODUCTION.md` for a ranked list of what a larger
  project would add and why.
- No synthetic-PDF generator. The three provided samples in `phase1_data/`
  are the authoritative test fixtures.
- No sanitizer layer that "fixes" OCR misreads. Rewriting `6512345678` to
  `0512345678` is invention, and the prompt explicitly forbids it.
