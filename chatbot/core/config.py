"""Runtime configuration for the chatbot service.

Extends the shared `AzureOpenAISettings` with the chatbot-specific fields
(knowledge base path, API server bind address, logging targets, and the
URL the Streamlit front-end uses to reach the back-end).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

from common.config import AzureOpenAISettings, PROJECT_ROOT

# Knowledge base lives at <project_root>/tests/test_data/phase2_data.
_KNOWLEDGE_BASE_DEFAULT = PROJECT_ROOT / "tests" / "test_data" / "phase2_data"


class ChatBotSettings(AzureOpenAISettings):
    # Longer timeout than Part 1 because the Q&A prompt may carry several
    # retrieved chunks plus conversation history.
    request_timeout_s: float = Field(default=90.0)

    # ── Knowledge base ─────────────────────────────────────────────────────────
    knowledge_base_path: Path = Field(default=_KNOWLEDGE_BASE_DEFAULT)

    # ── Retrieval (ADA-002) ────────────────────────────────────────────────────
    # Name of the Azure OpenAI *embedding* deployment (separate from the chat
    # deployment).  Matches the resource provided with the assignment.
    azure_openai_embedding_deployment: str = Field(default="text-embedding-ada-002")
    # When True the Q&A handler embeds the user's question, retrieves the
    # top-k most relevant chunks, and passes only those to GPT-4o.
    # Set to False to fall back to stuffing the full knowledge base into the
    # prompt (useful for A/B comparisons or when the embedding deployment is
    # unavailable).
    use_retrieval: bool = Field(default=True)
    # How many chunks to retrieve per Q&A turn.
    retrieval_top_k: int = Field(default=5)

    # ── API server ─────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="chatbot.log")

    # ── Frontend ───────────────────────────────────────────────────────────────
    # URL the Streamlit app uses to reach the FastAPI backend.
    api_base_url: str = Field(default="http://localhost:8000")


@lru_cache(maxsize=1)
def get_settings() -> ChatBotSettings:
    """Return a cached settings singleton."""
    return ChatBotSettings()
