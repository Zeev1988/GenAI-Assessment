"""Shared Azure OpenAI settings for both Part 1 and Part 2."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_FILE: Path = PROJECT_ROOT / ".env"


class AzureOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    azure_openai_endpoint: str = Field(default="")
    azure_openai_key: SecretStr = Field(default=SecretStr(""))
    azure_openai_api_version: str = Field(default="2024-12-01-preview")
    azure_openai_deployment: str = Field(default="gpt-4o")
    request_timeout_s: float = Field(default=30.0)
