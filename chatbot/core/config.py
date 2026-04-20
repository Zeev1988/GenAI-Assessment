"""Runtime configuration for the chatbot service.

Extends the shared `AzureOpenAISettings` with the chatbot-specific fields
(knowledge base path, API server bind address, logging targets, and the
URL the Streamlit front-end uses to reach the back-end).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

from common.config import PROJECT_ROOT, AzureOpenAISettings

# Knowledge base lives at <project_root>/tests/chatbot/test_data.
_KNOWLEDGE_BASE_DEFAULT = PROJECT_ROOT / "tests" / "chatbot" / "test_data"


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
