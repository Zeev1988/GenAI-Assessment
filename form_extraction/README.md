# Form 283 Extractor

Lean Part 1 solution for the GenAI take-home: upload a Bituach Leumi Form
283 (PDF/JPG/PNG), run Azure Document Intelligence for OCR, send the DI
Markdown output to GPT-4o with a strict JSON-schema structured output,
validate the result, and return the fields as JSON.

## Data flow

```
                 +--------------------+   DI Markdown    +------------------+
 PDF / JPG / PNG | Azure Document     |  ------------->  | GPT-4o           |
 --------------> |  Intelligence      |                  |  (json_schema,   |
                 |  prebuilt-layout   |                  |   strict mode)   |
                 |  (markdown output) |                  |                  |
                 +--------------------+                  +---------+--------+
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
                                                         |   JSON · MD · ⚠ |
                                                         +------------------+
```

## Layout

```
form_extraction/
  core/              # framework-agnostic library
    config.py        # env-loaded settings (pydantic-settings)
    schemas.py       # Pydantic models + Hebrew label map + OpenAI json_schema;
                     #   single source of truth for enum labels
    ocr.py           # Azure DI wrapper; returns OCRResult(markdown=...)
    prompts.py       # system prompt + one clean few-shot
    extractor.py     # one GPT-4o call on the Markdown + one retry on Pydantic
                     #   error + a deterministic digit-field fallback
    digits.py        # anchor-label parser for every numeric field on the
                     #   form (ID, 4 dates, 2 phones, postal, time, short
                     #   address digits); driven by a single registry
    validate.py      # format checks + anchor-disagreement warnings + grounding
    pipeline.py      # OCR -> extract -> validate
  ui/                # Streamlit entrypoint; imports only from core
    app.py
tests/form_extraction/
  test_schema.py     # round-trip + json_schema strictness
  test_validate.py   # format rules + grounding
  test_e2e.py        # parametrised E2E against the 3 known sample PDFs
                     #   (gated behind RUN_AZURE_TESTS=1; OCR results cached
                     #    under tests/form_extraction/fixtures/ocr_cache/)
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

Upload a PDF/JPG/PNG. The UI shows the extracted JSON, the DI Markdown,
and validation issues in three tabs. Toggle "Show Hebrew field labels"
to re-label the JSON with the Hebrew keys from the assignment spec;
both language variants are downloadable.

## Test

```bash
pytest                               # unit tests (no Azure calls)
pytest -m "not integration"          # explicitly skip the gated E2E tests
RUN_AZURE_TESTS=1 pytest             # include the live E2E tests
```

The E2E suite caches Azure DI output under
`tests/form_extraction/fixtures/ocr_cache/<stem>.json`. First run requires
`RUN_AZURE_TESTS=1` and live Azure DI + Azure OpenAI credentials;
subsequent runs skip Document Intelligence and hit only Azure OpenAI.

## Design notes

**One extractor, not two.** Azure DI's `prebuilt-layout` Markdown output
already handles reading order, renders tables with `|` cells, and emits
selection marks as `☒` / `☐` glyphs inline. GPT-4o reads that natively
and makes every extraction decision — names, dates, checkboxes, free
text. We deliberately do not run a second, coordinate-based extractor
for "the hard fields": it would either duplicate work the LLM already
does or introduce per-form-revision calibration that silently breaks
when the printer changes. One extractor with a principled prompt is
easier to reason about, test, and debug.

**Structured outputs carry the schema contract.** The extractor issues
one GPT-4o call with `response_format=json_schema` in strict mode, so
the response is guaranteed to match `ExtractedForm`'s JSON schema
(including the enum constraints on `gender`, `accidentLocation`, and
`healthFundMember`). One corrective re-ask runs only in the rare case
Pydantic still rejects the payload.

**Principled prompt + one clean few-shot.** The system prompt describes
what the form is, how DI renders it, and a short set of rules that
apply uniformly to every field: *copy verbatim*, *empty stays empty*,
*a ☒ selects the spatially-adjacent label; unresolvable → ""*, *dates
are all-or-none DDMMYYYY parts*, *address is either the street path
or the PO-Box path*. A single, normally-filled few-shot anchors the
OCR-to-JSON mapping for shapes that are hard to convey in prose. We
deliberately avoid enumerating sample-specific heuristics; those
over-fit to the three provided forms.

**Predictable `""` over silent guessing.** When the OCR reading order
makes a checkbox genuinely ambiguous — a ☒ floating alone, for example,
or candidate labels separated from the marker by a heading — the
prompt instructs the model to emit `""`. A reviewer can spot a missing
value in seconds; a plausible wrong value can live in the JSON for a
long time before anyone notices.

**Anchor-label parser, driven by one registry.** Every numeric field
on Form 283 is printed as a digit box preceded by a stable Hebrew
anchor label (`תאריך הפגיעה`, `ת.ז.`, `טלפון נייד`, `מיקוד`, ...).
`digits.py` has a single registry keyed by JSON field name; each
entry lists the field's anchors, acceptable digit counts, a
structural validator, and a few parser knobs (scan-window size, the
min-gap between digits of the box run). `parse_numeric(markdown,
field)` walks the registry: locate the anchor, carve a scan window
truncated at the next anchor so a noisy box can't latch onto the
neighbour's digits, run the box-pattern regex (digit, `\D{min,6}`,
digit, …), reverse for RTL context when Hebrew shares the digits'
line, apply the field's structural check, and return the first read
that passes. Dates (`parse_date`) and IDs (`parse_id`) remain as
thin wrappers so the extractor's date-tuple interface stays intact.

The registered fields and their structural checks are:

- `idNumber` — 9 or 10 digits.
- `dateOfBirth`, `dateOfInjury`, `formFillingDate`, `formReceiptDateAtClinic` — 8 digits, calendar-valid DDMMYYYY, 1900 ≤ year ≤ 2100.
- `mobilePhone` — 10 digits starting `05`.
- `landlinePhone` — 9 digits starting `0[234689]`.
- `postalCode` — 5 or 7 digits.
- `timeOfInjury` — 4 digits, HH in 00–23 and MM in 00–59.
- `apartment`, `entrance`, `houseNumber` — 1–4 digits (weak check; see below).

**Override vs warn-only.** Fields in the first six bullets have
structural checks strong enough that a validator-passing read is
worth trusting over the LLM's output, so `extractor.py` silently
replaces the LLM value when the parser speaks. This is why the
override is safe despite being unconditional: on displaced-anchor
cases (ex2 `idNumber`, ex3 `dateOfBirth`) the parser returns `None`
and the LLM's value survives — the parser only speaks when every
check has passed. Short address fields have a weak "1–4 digits"
check that a stray digit run in the scan window can pass by
coincidence, so they are `override=False`: `validate.py` compares
the parser's read against the LLM's value and surfaces a warning on
disagreement, and the JSON is left alone. The reviewer decides.

Anchor labels are preferred over pixel coordinates for the same
reason the rest of the extractor is OCR-text-based: anchors ride
along with any form revision that keeps the same field names,
whereas coordinate polygons need per-revision calibration and
silently break when printer or scan margins change.

**RTL bidi reversal.** Azure DI emits characters in logical order. On a
line that starts with Hebrew letters (an RTL run), digits following
them are written to the logical stream in *visual right-to-left*
order — the opposite of how a human reads the box. ex3's ID box
(`עי 7 6 5 1| 2 5 | 4 3 3 | 0` → digits `7651254330` → actual
`0334521567`) is the concrete case. The parser detects the situation
structurally (any Hebrew character on the same line before the match)
and reverses the digit sequence before structural validation. The same
rule applies uniformly to every digit field — IDs, dates, anything —
so we never need field-specific reversal knobs.

**Validation is observational.** `validate.py` runs (a) format rules
(ID length, phone shape, HH:MM time, real calendar dates), (b) an
*anchor-disagreement* check that flags warn-only numeric fields
(apartment, entrance, house number) where the anchor-label parser's
read differs from the LLM's value, and (c) a grounding check that
flags free-text values which don't appear as substrings of the OCR
Markdown. None of the layers rewrite values; the UI surfaces issues
and the reviewer decides. Enum fields are not grounded (they come
from checkbox resolution, not OCR body prose).

**Single source of truth for enum labels.** `schemas.py` owns the
`GENDER_LABELS`, `HEALTH_FUND_LABELS`, and `ACCIDENT_LOCATION_LABELS`
tuples; the prompt enum lists, the JSON-schema `enum` arrays, and the
test assertions all derive from them. Prompt and strict response
format cannot drift out of sync.

**Accuracy check is human-in-the-loop.** The UI's "OCR Markdown" tab
shows the DI output next to the extracted JSON so the reviewer can
eyeball correctness — still the most reliable accuracy signal, and the
thing the assignment explicitly asks for.

**Missing fields are empty strings** per the spec. The Pydantic models
default every field to `""`, and the prompt instructs the model to
leave them that way when a value is not clearly filled.

**Hebrew and English** are both handled by keeping the OCR content in
its native script and instructing the LLM never to translate values.
The UI toggle relabels the keys of the JSON output into Hebrew without
touching the values.

**PII-safe structured logging** via stdlib `logging`. The UI sets up
`basicConfig` at INFO; each pipeline stage logs a `name.event key=value`
line with byte counts, char counts, elapsed ms, and field counts. Raw
field values are logged only at DEBUG level.

## Known limitations

- When OCR reading order fragments a Section 5 checkbox (a ☒ floating
  between or away from its intended fund label), a principled extractor
  emits `""` rather than guessing. This is deliberate: a missed answer
  is fixable with one click in the UI; an invented answer is not.
- The grounding check is substring-based; a hallucination that happens
  to collide with text elsewhere on the form (e.g. a first name pulled
  into a diagnosis field) won't be flagged. The prompt is the primary
  defence; grounding is the backstop.
- The anchor-label digit parser cannot recover fields whose OCR reading
  order displaces the anchor from its box. On the sample data this
  affects ex2 `idNumber` (anchor intact, box digits displaced into an
  unrelated area) and ex3 `dateOfBirth` (anchor appears *after* its
  box, with injury-date free text in between). The parser safely
  returns `None` in both cases and the LLM's value is kept; we do not
  attempt to guess.
- Strict `json_schema` mode forces every property to be present. The
  prompt's "empty stays empty" rule produces `""` for absent values.
- Azure Document Intelligence accuracy degrades on scans rotated >15°.
  The prebuilt-layout model is the best we get without a custom model.

## Not included on purpose

- No persistent cache, correlation IDs, PII redactor, or retry wrapper.
  The brief is a prototype, not a production microservice.
- No synthetic-PDF generator. The three provided samples are the
  authoritative test fixtures.
- No sanitiser layer that "fixes" OCR misreads. Rewriting `6512345678`
  to `0512345678` is invention, and the prompt explicitly forbids it.
- No signature image capture. The schema keeps a `signature` key for
  parity with the assignment spec but it is always `""` — the
  applicant's name lives in `firstName`/`lastName`.
