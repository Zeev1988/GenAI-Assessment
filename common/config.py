"""Application configuration, sourced from environment variables.

All secrets are read via ``pydantic-settings`` so they never need to appear in
source. Values can be provided in a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    azure_doc_intelligence_endpoint: str = Field(
        default="",
        description="Endpoint URL for Azure AI Document Intelligence.",
    )
    azure_doc_intelligence_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Azure AI Document Intelligence.",
    )
    azure_doc_intelligence_api_version: str = Field(default="2024-11-30")

    azure_openai_endpoint: str = Field(default="")
    azure_openai_key: SecretStr = Field(default=SecretStr(""))
    azure_openai_api_version: str = Field(default="2024-10-21")
    azure_openai_deployment_extract: str = Field(default="gpt-4o")
    azure_openai_deployment_judge: str = Field(default="gpt-4o-mini")
    azure_openai_deployment_chat: str = Field(default="gpt-4o")
    azure_openai_deployment_embedding: str = Field(default="text-embedding-ada-002")

    app_log_level: str = Field(default="INFO")
    app_cache_dir: str = Field(default=".cache")
    app_max_upload_mb: int = Field(default=10, ge=1, le=100)
    app_request_timeout_s: float = Field(default=90.0, gt=0.0)
    app_enable_llm_judge: bool = Field(default=False)

    @property
    def cache_path(self) -> Path:
        return Path(self.app_cache_dir).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.app_max_upload_mb * 1024 * 1024

    def require_azure_di(self) -> None:
        """Raise if Document Intelligence credentials are missing."""
        if (
            not self.azure_doc_intelligence_endpoint
            or not self.azure_doc_intelligence_key.get_secret_value()
        ):
            raise RuntimeError(
                "Azure Document Intelligence credentials are not configured. "
                "Set AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY in .env."
            )

    def require_azure_openai(self) -> None:
        """Raise if Azure OpenAI credentials are missing."""
        if not self.azure_openai_endpoint or not self.azure_openai_key.get_secret_value():
            raise RuntimeError(
                "Azure OpenAI credentials are not configured. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY in .env."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (safe for concurrent use)."""
    return Settings()
