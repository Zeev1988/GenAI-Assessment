# Medical Chatbot (Phase 2)

Stateless microservice-based chatbot answering questions about medical
services for Israeli health funds (מכבי / מאוחדת / כללית), backed by the
HTML knowledge base under `phase2_data/`.

This package will contain:

- `backend/` - FastAPI stateless microservice (no server-side user memory).
  Endpoints for (1) user-information collection phase (LLM-driven) and
  (2) Q&A phase (RAG over the HTML knowledge base). All conversation state
  is round-tripped via the request body.
- `frontend/` - Streamlit or Gradio chat UI driving the backend.
- `knowledge_base/` - Parsers and retrievers over `phase2_data/*.html`.

It reuses the shared `common/` package for configuration, Azure OpenAI
client, retries, structured logging, and caching.

> Not implemented yet. See the main `README.md` for the Phase 1 deliverable.
