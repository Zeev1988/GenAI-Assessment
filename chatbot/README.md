# HMO Chatbot

**Part 2 of the GenAI take-home:** A stateless FastAPI chatbot that answers questions about Israeli health-fund (קופות חולים) services for Maccabi, Meuhedet, and Clalit members.

It collects the user's details via LLM tool-calling, then answers benefit questions using ADA-002 retrieval over a chunked HTML knowledge base.

---

## Architecture

```text
┌───────────────────────────────────────────────────────┐
│  Streamlit UI  (chatbot/ui/app.py)     port 8501      │
│  State is client-side; full history sent each turn    │
└───────────────────┬───────────────────────────────────┘
                    │ POST /api/v1/chat
                    ▼
┌───────────────────────────────────────────────────────┐
│  FastAPI backend  (chatbot/api/main.py)  port 8000    │
│  • "collection": LLM gathers member info via tool     │
│  • "qa":  ADA-002 retrieval → top-k chunks → GPT-4o   │
└───────────────────────────────────────────────────────┘
```

---

## Quick Start

Requires **Python 3.11+**.

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in AZURE_OPENAI_* keys
```

Launch the two services:

```bash
python chatbot/run_chatbot.py api    # or: python -m chatbot.api.main
python chatbot/run_chatbot.py ui     # or: python -m streamlit run chatbot/ui/app.py
```

Interactive docs at `http://localhost:8000/docs`, UI at `http://localhost:8501`.

---

## Tests

```bash
curl http://localhost:8000/health
pytest tests/chatbot/
```

See [DESIGN.md](./DESIGN.md) for the retrieval strategy and design trade-offs.
