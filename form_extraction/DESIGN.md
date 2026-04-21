# Design Notes — Form 283 Extractor

## One extractor, not two

Azure DI's `prebuilt-layout` Markdown output handles reading order, renders tables with `|` cells, and emits selection marks as `☒`/`☐` glyphs. GPT-4o reads that directly. We avoided a second coordinate-based extractor because it would duplicate work and needs re-calibration whenever printer margins shift.

## Structured outputs

The extractor issues one GPT-4o call with `response_format=json_schema` in strict mode. The response is guaranteed to match the schema (including enum constraints). A corrective re-ask runs only when Pydantic rejects the payload (rare).

## Prompt

One system prompt describing the form and uniform reasoning rules (*copy verbatim*, *empty stays empty*, *☒ selects the adjacent label*) plus one clean few-shot. We avoided sample-specific heuristics.

## Anchor-label digit parser (`digits.py`)

Every numeric field on Form 283 sits next to a stable Hebrew anchor (`ת.ז.`, `תאריך הפגיעה`, etc.). The parser locates the anchor, carves a scan window (truncated at the next known anchor), runs a box-pattern regex, reverses the digits if the line starts with Hebrew (RTL visual order), and validates structurally.

Fields with strong structural checks (ID, dates, phones, HH:MM, 5/7-digit postal) use `override=True`: the parser's read replaces the LLM's value silently. Short fields (`apartment`, `entrance`, `houseNumber`) have weaker checks, so they use `override=False` — `validate.py` surfaces a warning on disagreement and the LLM value is kept.

## Validation (`validate.py`)

Non-destructive checks:
1. **Format rules:** ID length, phone prefixes, calendar-valid dates, HH:MM.
2. **Anchor disagreements:** warn when the parser disagrees with the LLM on a warn-only field.
3. **Grounding:** warn when a free-text value does not appear as a substring of the OCR.

None of these rewrite the JSON; the UI surfaces them for review.

## Enum source of truth

`schemas.py` owns the gender / health-fund / accident-location label tuples. The prompt's allowed-value lists, the JSON schema `enum` arrays, and the tests all derive from these tuples so they cannot drift apart.
