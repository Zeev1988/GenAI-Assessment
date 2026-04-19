"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root:
# form_extraction/core/config.py → form_extraction/core → form_extraction → project root
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    azure_doc_intelligence_endpoint: str = Field(default="")
    azure_doc_intelligence_key: SecretStr = Field(default=SecretStr(""))

    azure_openai_endpoint: str = Field(default="")
    azure_openai_key: SecretStr = Field(default=SecretStr(""))
    azure_openai_api_version: str = Field(default="2024-12-01-preview")
    azure_openai_deployment: str = Field(default="gpt-4o")

    # Network timeout (seconds) applied to both Azure clients.
    request_timeout_s: float = Field(default=30.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
