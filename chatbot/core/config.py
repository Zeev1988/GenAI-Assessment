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
    # Longer timeout than Part 1 because the full knowledge base is passed
    # on every Q&A turn (the LLM's prompt can be 20k+ tokens).
    request_timeout_s: float = Field(default=90.0)

    # ── Knowledge base ─────────────────────────────────────────────────────────
    knowledge_base_path: Path = Field(default=_KNOWLEDGE_BASE_DEFAULT)

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
