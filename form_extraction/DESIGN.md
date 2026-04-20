# Architecture & Design Notes

This document details the engineering decisions, trade-offs, and parsing logic behind the Form 283 Extractor.

---

## 🧠 Core Decisions

### One Extractor, Not Two
Azure DI's `prebuilt-layout` Markdown output already handles reading order, renders tables with `|` cells, and emits selection marks as `☒` / `☐` glyphs inline. GPT-4o reads this natively and makes every extraction decision. We deliberately avoided running a second, coordinate-based extractor for "hard fields" because it duplicates LLM work and introduces per-form-revision calibration that silently breaks when printer margins change. 

### Structured Outputs Carry the Schema Contract
The extractor issues one GPT-4o call with `response_format=json_schema` in strict mode. The response is mathematically guaranteed to match `ExtractedForm`'s JSON schema (including `enum` constraints). A corrective re-ask runs only in the rare case Pydantic rejects the payload.

### Principled Prompts + One Clean Few-Shot
The system prompt describes how DI renders the form and provides uniform rules: *copy verbatim*, *empty stays empty*, *a ☒ selects the spatially-adjacent label*, etc. A single, normally-filled few-shot anchors the OCR-to-JSON mapping. We avoided sample-specific heuristics that over-fit to the three provided test forms.

### Predictable `""` Over Silent Guessing
When the OCR reading order makes a checkbox genuinely ambiguous (e.g., a `☒` floating alone), the prompt instructs the model to emit `""`. A human reviewer can spot a missing value in seconds; a plausible but wrong value can live in the database forever.

---

## 🔢 The Anchor-Label Parser (`digits.py`)

Every numeric field on Form 283 is printed as a digit box preceded by a stable Hebrew anchor label (e.g., `תאריך הפגיעה`, `ת.ז.`). 

`digits.py` uses a single registry keyed by JSON field name. `parse_numeric(markdown, field)` locates the anchor, carves a scan window (truncated at the next anchor), runs a box-pattern regex, reverses for RTL context (if needed), applies structural checks, and returns the read.

### Structural Checks
* **`idNumber`**: 9 or 10 digits.
* **Dates** (`dateOfBirth`, `dateOfInjury`, etc.): 8 digits, calendar-valid DDMMYYYY, 1900 ≤ year ≤ 2100.
* **Phones**: `mobilePhone` (10 digits starting `05`), `landlinePhone` (9 digits starting `0[234689]`).
* **`postalCode`**: 5 or 7 digits.
* **`timeOfInjury`**: 4 digits, valid HH:MM.
* **Short Addresses** (`apartment`, `entrance`): 1–4 digits.

### Override vs. Warn-Only
Fields with strong structural checks (IDs, Dates, Phones) are trusted over the LLM's output. `extractor.py` **silently replaces** the LLM value when the parser succeeds. Short address fields have a weak "1–4 digits" check, so they are set to `override=False`: `validate.py` simply compares the parser's read against the LLM's value and surfaces a UI warning on disagreement.

### RTL Bidi Reversal
Azure DI emits characters in logical order. On a line starting with Hebrew letters, digits following them are written in *visual right-to-left* order. The parser detects this structurally (any Hebrew character on the same line before the match) and automatically reverses the digit sequence before validation.

---

## 🛡️ Validation & Safety

### Observational Validation
`validate.py` runs non-destructive checks:
1.  **Format Rules:** ID length, real calendar dates, etc.
2.  **Anchor Disagreements:** Flags warn-only numeric fields where the parser differs from the LLM.
3.  **Grounding Check:** Flags free-text values that do not appear as exact substrings in the OCR Markdown. 

*None of these rewrite values; the UI surfaces the issues for the human reviewer.*

### Single Source of Truth for Enums
`schemas.py` owns the `GENDER_LABELS`, `HEALTH_FUND_LABELS`, etc. The prompt enum lists, the JSON-schema `enum` arrays, and the test assertions all derive from these exact tuples so they cannot drift out of sync.

### Human-in-the-Loop Verification
The UI's "OCR Markdown" tab shows the DI output next to the extracted JSON so the reviewer can eyeball correctness — the most reliable accuracy signal.

### PII-Safe Structured Logging
Via the stdlib `logging`. Each pipeline stage logs a `name.event key=value` line with byte counts, elapsed ms, and field counts. Raw, potentially sensitive field values are strictly reserved for `DEBUG` level.

---

## ⚠️ Known Limitations

* **Fragmented Checkboxes:** When OCR fragments a Section 5 checkbox away from its label, the extractor emits `""`. This is deliberate to prevent hallucinations.
* **Substring Grounding:** The grounding check won't catch a hallucination if the hallucinated text happens to collide with valid text elsewhere on the form.
* **Displaced Anchors:** The digit parser cannot recover fields where OCR displaces the anchor from its box. The parser safely returns `None`, and the LLM's value is kept.

## 🚫 Not Included On Purpose

* No persistent cache, PII redactor, or retry wrappers (this is a prototype).
* No synthetic-PDF generators (the three provided samples are the authoritative tests).
* No sanitiser layer that "fixes" OCR misreads (rewriting `6512345678` to `0512345678` is invention; the prompt forbids it).