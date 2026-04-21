# Form 283 Extractor

**Part 1 of the GenAI take-home:** A pipeline that extracts Bituach Leumi Form 283 (PDF/JPG/PNG) into validated JSON.

---

## Flow

```text
PDF/JPG/PNG → Azure Document Intelligence (prebuilt-layout, markdown)
            → GPT-4o (strict json_schema)
            → Pydantic ExtractedForm
            → validate() formats + grounding
            → Streamlit UI
```

---

## Quick Start

Requires **Python 3.11+**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # fill in AZURE_OPENAI_* and AZURE_DOCUMENT_INTELLIGENCE_*
streamlit run form_extraction/ui/app.py
```

---

## Tests

```bash
pytest                               # unit tests (no Azure calls)
RUN_AZURE_TESTS=1 pytest             # live E2E against sample PDFs (cached)
```

See [DESIGN.md](./DESIGN.md) for design trade-offs and the anchor-label parser details.
