# GenAI Assessment

Two-part take-home for Bituach Leumi (Israeli National Insurance) GenAI tooling, both backed by Azure OpenAI.

---

## Parts

### Part 1 — Form 283 Extractor (`form_extraction/`)

Uploads a Bituach Leumi Form 283 (PDF/JPG/PNG), runs Azure Document Intelligence for coordinate-based OCR, then calls GPT-4o with a strict JSON schema to extract all fields.

**Stack:** Azure Document Intelligence · Azure OpenAI GPT-4o · Pydantic · Streamlit

```bash
python -m streamlit run form_extraction/ui/app.py
```

→ See [`form_extraction/README.md`](form_extraction/README.md) for full details.

---

### Part 2 — HMO Chatbot (`chatbot/`)

A stateless FastAPI chatbot that answers questions about Israeli health-fund (קופות חולים) services for Maccabi, Meuhedet, and Clalit members. Collects member info via LLM tool-calling, then answers benefit questions using ADA-002 retrieval over a chunked HTML knowledge base.

**Stack:** Azure OpenAI GPT-4o · ADA-002 embeddings · FastAPI · Streamlit

```bash
# Terminal 1 — API
python -m chatbot.api.main

# Terminal 2 — UI
streamlit run chatbot/ui/app.py
```

→ See [`chatbot/README.md`](chatbot/README.md) for full details.

---

## Shared infrastructure (`common/`)

Both parts share `common/config.py` (Pydantic-settings for Azure OpenAI credentials) and `common/logger.py` (rotating file + console logger).

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # fill in AZURE_* values
```

**.env keys required:**

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Part 1 only
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=<your-key>
```

---

## Tests

```bash
pytest                       # all unit tests (no Azure calls)
RUN_AZURE_TESTS=1 pytest     # include live E2E tests (requires credentials)
```
