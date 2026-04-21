"""Chatbot settings — extends the shared AzureOpenAISettings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

from common.config import PROJECT_ROOT, AzureOpenAISettings

_KNOWLEDGE_BASE_DEFAULT = PROJECT_ROOT / "tests" / "chatbot" / "test_data"


class ChatBotSettings(AzureOpenAISettings):
    request_timeout_s: float = Field(default=90.0)

    knowledge_base_path: Path = Field(default=_KNOWLEDGE_BASE_DEFAULT)

    azure_openai_embedding_deployment: str = Field(default="text-embedding-ada-002")
    retrieval_top_k: int = Field(default=5)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    log_level: str = Field(default="INFO")
    log_file: str = Field(default="chatbot.log")

    api_base_url: str = Field(default="http://localhost:8000")


@lru_cache(maxsize=1)
def get_settings() -> ChatBotSettings:
    return ChatBotSettings()
