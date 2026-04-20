# Part 2 — Microservice-based HMO Chatbot

A stateless chatbot that answers questions about Israeli health-fund (קופות חולים) services for **Maccabi**, **Meuhedet**, and **Clalit** members.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  Streamlit UI  (chatbot/ui/app.py)          port 8501         │
│                                                               │
│  • All state stored in st.session_state (client-side)         │
│  • Sends full conversation history + user_info on every turn  │
└───────────────────┬───────────────────────────────────────────┘
                    │ HTTP POST /api/v1/chat
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  FastAPI backend  (chatbot/api/main.py)     port 8000         │
│                                                               │
│  • Fully stateless — no server-side session storage           │
│  • Phase "collection": LLM gathers member info via tool-call  │
│  • Phase "qa":         LLM answers using HTML knowledge base  │
└───────────────────┬───────────────────────────────────────────┘
                    │ Azure OpenAI API (GPT-4o)
                    ▼
┌───────────────────────────────────────────────────────────────┐
│  Azure OpenAI  (GPT-4o)                                       │
│  Knowledge base: tests/test_data/phase2_data/*.html           │
└───────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Concern | Solution |
|---|---|
| Statelessness | Full conversation history + user info sent on every request |
| Phase transition | LLM calls `submit_user_info` tool when collection is confirmed |
| Concurrency | FastAPI async + single shared Azure OpenAI client |
| Streamlit + async | Synchronous `requests` in UI threads — each user session is isolated |
| Multi-language | LLM detects Hebrew/English per-message and mirrors it |
| Knowledge base | Raw HTML passed to GPT-4o (understands HTML natively) |

---

## Directory layout

```
chatbot/
├── api/
│   ├── __init__.py
│   └── main.py          FastAPI app, Pydantic models, route handlers
├── core/
│   ├── __init__.py
│   ├── config.py        Pydantic-settings (reads project-root .env)
│   ├── knowledge.py     HTML knowledge-base loader (singleton)
│   └── prompts.py       System prompts + submit_user_info tool schema
├── ui/
│   ├── __init__.py
│   └── app.py           Streamlit frontend (all state in session_state)
└── README.md            (this file)
```

---

## Quick start

### 1 — Install dependencies

```bash
# From the project root
pip install -e ".[dev]"
```

This installs Part 1 (`form_extraction`), Part 2 (`chatbot`), and all dev
tools from the root `pyproject.toml`.

### 2 — Set credentials

The chatbot reuses the same `.env` file as Part 1.  Make sure it contains:

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Optional overrides:

```dotenv
CHATBOT_API_URL=http://localhost:8000   # URL the Streamlit app uses to reach the API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
LOG_FILE=chatbot.log
```

### 3 — Start the API server

```bash
# From the project root
python -m chatbot.api.main
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 4 — Start the Streamlit frontend

Open a **second terminal** in the project root:

```bash
streamlit run chatbot/ui/app.py
```

The UI will open at `http://localhost:8501`.

---

## API reference

### `GET /health`

Returns service status and knowledge-base readiness.

```json
{
  "status": "ok",
  "knowledge_base_loaded": true,
  "topic_count": 6,
  "topics": ["רפואה משלימה / Alternative Medicine", "..."],
  "deployment": "gpt-4o"
}
```

### `POST /api/v1/chat`

**Request body**

```json
{
  "phase": "collection",
  "messages": [
    {"role": "user",      "content": "שלום"},
    {"role": "assistant", "content": "שלום! שמי עוזר..."},
    {"role": "user",      "content": "ישראל ישראלי"}
  ],
  "user_info": null,
  "request_id": "a1b2c3d4-..."
}
```

| Field | Description |
|---|---|
| `phase` | `"collection"` or `"qa"` |
| `messages` | Full conversation history including the latest user message.  May be empty on first load. |
| `user_info` | `null` during collection; confirmed member dict during Q&A |
| `request_id` | Optional UUID for tracing (auto-generated if omitted) |

**Response body**

```json
{
  "message":             "תודה! כל הפרטים אושרו...",
  "phase":               "qa",
  "transition":          true,
  "extracted_user_info": {
    "first_name": "ישראל",
    "last_name": "ישראלי",
    "id_number": "123456789",
    "gender": "זכר",
    "age": 35,
    "hmo_name": "מכבי",
    "hmo_card_number": "987654321",
    "insurance_tier": "זהב"
  },
  "request_id":          "a1b2c3d4-...",
  "processing_time_ms":  1234
}
```

---

## User information collected

| Field | Validation |
|---|---|
| First name | non-empty string |
| Last name | non-empty string |
| ID number | exactly 9 digits |
| Gender | זכר / נקבה / אחר |
| Age | integer 0–120 |
| HMO name | מכבי / מאוחדת / כללית |
| HMO card number | exactly 9 digits |
| Insurance tier | זהב / כסף / ארד |

All validation is handled by the LLM during conversation — no hardcoded form logic.

---

## Knowledge base

Six HTML files in `tests/test_data/phase2_data/`:

| File | Service category |
|---|---|
| `alternative_services.html` | רפואה משלימה |
| `communication_clinic_services.html` | מרפאות תקשורת |
| `dentel_services.html` | מרפאות שיניים |
| `optometry_services.html` | אופטומטריה |
| `pragrency_services.html` | הריון |
| `workshops_services.html` | סדנאות בריאות |

---

## Design choices

**Why no ADA-002 embeddings / retrieval?** The knowledge base is six short
HTML files (~20–30 KB total). The full content fits comfortably in GPT-4o's
context window with room to spare for the system prompt, conversation
history, and the answer. Embedding + top-k retrieval would add latency,
operational cost, and a failure mode (irrelevant chunks leaking into the
context) without measurably improving answer quality at this size.

The decision **would flip** once the knowledge base exceeded roughly 50 KB
or started covering substantially more topics, at which point the marginal
cost of a larger prompt outweighs the operational simplicity.  The
retrieval layer would then be a simple `embeddings.create()` call at
startup to index every document, plus a cosine-similarity lookup per
request — the rest of the pipeline would not need to change.

**Why tool-calling for the phase transition?** Using `submit_user_info` as
an OpenAI function-call lets the LLM signal "collection complete" with a
strongly-typed payload (the tool parameters are a JSON schema with `enum`
constraints for gender / HMO / tier). The alternative — a keyword like
`[DONE]` in the reply text, or a separate classifier call — is both less
reliable and not self-documenting. The schema *is* the contract.

**Why pass the full conversation history on every request?** The brief
mandates statelessness, but there's also a pragmatic reason: any
server-side session store introduces sticky routing and a new failure mode
for multi-instance deployments. Passing history client-side means the API
is infinitely horizontally scalable and a Container App restart can't lose
any user's context.

---

## Logging

Logs go to both the console and `chatbot.log` (rotating, 5 MB × 3 files).
Every request is logged with its `request_id`, phase, token usage, and
response time.  The logger factory lives in the shared `common/logger.py`
and is also used by Part 1.

---

## Running tests

```bash
# Health check (API must be running)
curl http://localhost:8000/health

# Example chat request
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "collection",
    "messages": [],
    "user_info": null
  }'
```
