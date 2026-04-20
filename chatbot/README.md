# HMO Chatbot

**Part 2 of the GenAI take-home:** A stateless FastAPI chatbot that answers questions about Israeli health-fund (קופות חולים) services for Maccabi, Meuhedet, and Clalit members. 

It collects user information via LLM tool-calling and answers benefit questions using ADA-002 retrieval over a chunked HTML knowledge base.

---

## 🌊 Architecture Flow

```text
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
```

---

## 🚀 Quick Start

Requires **Python 3.11+**.

### 1. Install & Configure
```bash
# Install dependencies from the project root
pip install -e ".[dev]"

# Setup environment variables (shared with Part 1)
cp .env.example .env
# ⚠️ Ensure you fill in AZURE_OPENAI_* keys
```

### 2. Launch the Services
You can run the API and UI in two separate terminals, or use the included convenience launcher:

```bash
# Option A: Run both simultaneously in the background/foreground
python chatbot/run_chatbot.py both

# Option B: Run individually
python chatbot/run_chatbot.py api
python chatbot/run_chatbot.py ui
```
*The API interactive docs will be at `http://localhost:8000/docs` and the Chat UI at `http://localhost:8501`.*

---

## 🧪 Testing

```bash
# Verify API health (API must be running)
curl http://localhost:8000/health

# Run test suite
pytest tests/chatbot/
```

---

## ✨ System Highlights

* **Stateless API:** Full conversation history and user context are passed on every request, making the backend infinitely horizontally scalable.
* **Tool-Calling Router:** Uses an explicit `submit_user_info` OpenAI function-call to strictly enforce the phase transition between data collection and Q&A.
* **Semantic HTML Chunking:** The knowledge base is dynamically chunked by table row (`<tr>`) to preserve column-to-HMO alignment perfectly.
* **In-Memory Retrieval:** Uses a blazing-fast linear cosine scan over ADA-002 vectors in memory, avoiding heavy vector DB dependencies for a micro-corpus.

---

📖 **For a deep dive into the retrieval strategy, why we didn't use LangChain/VectorDBs, and prompt engineering trade-offs, please read [DESIGN.md](./DESIGN.md).**
