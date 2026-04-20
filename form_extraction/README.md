# Form 283 Extractor

**Part 1 of the GenAI take-home:** A pipeline that extracts Bituach Leumi Form 283 (PDF/JPG/PNG) into strictly typed, validated JSON.

---

## 🌊 Data Flow

```text
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

---

## 🚀 Quick Start

Requires **Python 3.11+**.

### 1. Install & Configure
```bash
# Install dependencies from the project root
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Setup environment variables
cp .env.example .env
# ⚠️ Ensure you fill in AZURE_OPENAI_* and AZURE_DOCUMENT_INTELLIGENCE_* keys
```

### 2. Launch the UI
```bash
streamlit run form_extraction/ui/app.py
```
*Upload a file in the UI to see the extracted JSON alongside the intermediate OCR Markdown and validation warnings.*

---

## 🧪 Testing

```bash
pytest                               # Unit tests (no live Azure calls)
RUN_AZURE_TESTS=1 pytest             # Live E2E tests against sample PDFs
```
*> **Note:** E2E tests automatically cache Azure DI output locally to save time and API costs on subsequent runs.*

---

## ✨ System Highlights

* **Single Extractor Pipeline:** Relies natively on Azure DI's reading order and markdown generation rather than building brittle, coordinate-based bounding boxes.
* **Strict Output:** GPT-4o is constrained by a strict JSON schema derived directly from Pydantic models.
* **Deterministic Fallbacks:** For critical numeric fields (ID, Dates, Phone numbers), a custom anchor-label parser (`digits.py`) validates and overrides the LLM's extraction.
* **RTL Bidi Support:** Automatically reverses digit sequences on lines starting with Hebrew characters to correct visual right-to-left OCR anomalies.

---

📖 **For a deep dive into the architectural trade-offs, OCR anchoring logic, and edge cases, please read [DESIGN.md](./DESIGN.md).**
