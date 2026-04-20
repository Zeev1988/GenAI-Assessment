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
│  • Phase "qa":  ADA-002 retrieval → top-k chunks → GPT-4o     │
└───────────────────┬───────────────────────────────────────────┘
                    │
            ┌───────┴────────┐
            ▼                ▼
  ┌─────────────────┐  ┌─────────────────┐
  │ Azure OpenAI    │  │ Azure OpenAI    │
  │ text-embedding- │  │ GPT-4o          │
  │ ada-002         │  │ (chat)          │
  │ (indexing +     │  │                 │
  │  per-query)     │  │                 │
  └─────────────────┘  └─────────────────┘

  Knowledge base: tests/chatbot/test_data/*.html
  → indexed at startup as ~42 chunks (1 intro + 1 per table row, per file)
```

### Key design decisions

| Concern | Solution |
|---|---|
| Statelessness | Full conversation history + user info sent on every request |
| Phase transition | LLM calls `submit_user_info` tool when collection is confirmed |
| Retrieval | ADA-002 embeddings + in-memory cosine top-k (no vector DB) |
| Concurrency | FastAPI async + single shared Azure OpenAI client |
| Streamlit + async | Synchronous `requests` in UI threads — each user session is isolated |
| Multi-language | LLM detects Hebrew/English per-message and mirrors it |
| Knowledge base | Chunked by table row; raw HTML passed to GPT-4o for structure |

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
│   ├── knowledge.py     HTML knowledge-base loader + chunker hook
│   ├── retrieval.py     HTML chunker + ADA-002 Retriever (in-memory top-k)
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
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

Optional overrides:

```dotenv
CHATBOT_API_URL=http://localhost:8000   # URL the Streamlit app uses to reach the API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
LOG_FILE=chatbot.log

# How many chunks to send to GPT-4o per question (see "Retrieval layer").
RETRIEVAL_TOP_K=5
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

Returns service status, knowledge-base readiness, and retrieval status.

```json
{
  "status": "ok",
  "knowledge_base_loaded": true,
  "topic_count": 6,
  "topics": ["רפואה משלימה / Alternative Medicine", "..."],
  "deployment": "gpt-4o",
  "retrieval": {
    "ready": true,
    "indexed_chunks": 42,
    "top_k": 5,
    "embedding_deployment": "text-embedding-ada-002"
  }
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

Six HTML files in `tests/chatbot/test_data/`:

| File | Service category |
|---|---|
| `alternative_services.html` | רפואה משלימה |
| `communication_clinic_services.html` | מרפאות תקשורת |
| `dentel_services.html` | מרפאות שיניים |
| `optometry_services.html` | אופטומטריה |
| `pragrency_services.html` | הריון |
| `workshops_services.html` | סדנאות בריאות |

Each file is one ``<h2>`` topic plus a table where every ``<tr>`` row is
one service with columns per HMO (מכבי, מאוחדת, כללית) and benefits per
tier (זהב, כסף, ארד).

---

## Retrieval layer

At startup the backend splits every HTML file into chunks — one chunk for
the intro narrative and one chunk per service table row — then embeds each
chunk with **Azure OpenAI `text-embedding-ada-002`** and stores the
vectors in memory.  The total corpus is ~42 chunks at ~430 characters
each.

On every Q&A request the backend:

1. Extracts the latest user message.
2. Embeds it (ADA-002, one call).
3. Ranks all chunks by cosine similarity.
4. Sends the top `RETRIEVAL_TOP_K` chunks (default 5) to GPT-4o.

Each chunk carries its parent topic title, so a single "80% הנחה" row
stays grounded in its service category when injected into the prompt.

### Why retrieval (instead of stuffing the full KB)?

The six HTML files total ~43 KB — comfortably inside GPT-4o's context
window — so prompt-stuffing would technically work.  Retrieval is used
anyway because it's the production shape of the system: a real HMO
service catalogue is thousands of rows, and the retrieval path is what
scales.  Narrower context also measurably reduces mix-ups between tiers
and HMOs on spot-checks.  The implementation cost is trivial: one
embedding call per turn and a linear cosine scan over ~42 vectors, which
is faster than the network round-trip to GPT-4o.

### Why no vector DB (faiss / chroma / pgvector)?

A linear cosine scan over ~42 vectors is in the microsecond range, so a
vector database would only add a dependency and a persistence story in
exchange for performance the user cannot perceive.  The assignment's "no
LangChain / no frameworks" rule reinforces this: the native Azure OpenAI
SDK is the only network dependency, and `math.sqrt` does the rest.

---

## Other design choices

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

**Why chunk by `<tr>` instead of by size?** The service rows *are* the
semantic units of this corpus — each row answers a single "how much does
X cost at tier Y for HMO Z?" question.  A fixed-size chunker would cut
through table cells and lose the column-to-HMO alignment.  Chunking on
structural boundaries preserves that alignment and makes every chunk
self-contained.

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
