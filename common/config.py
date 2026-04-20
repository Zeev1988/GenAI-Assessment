"""Shared settings base for Part 1 and Part 2.

Both sub-projects load the same repo-root `.env` file and reuse the same
Azure OpenAI credentials, so that configuration lives here exactly once.
Each sub-project extends `AzureOpenAISettings` with its own specific fields
(Document Intelligence for Part 1, knowledge-base path + server settings
for Part 2).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# common/config.py lives at <project_root>/common/config.py,
# so the project root is parents[1].
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ENV_FILE: Path = PROJECT_ROOT / ".env"


class AzureOpenAISettings(BaseSettings):
    """Base settings class with Azure OpenAI credentials + env-file loading.

    Sub-classes should add any project-specific fields and, if needed,
    override `request_timeout_s` (e.g. the chatbot needs a longer timeout
    because it passes the full knowledge base on every Q&A request).
    """

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

    # Default network timeout (seconds) applied to the Azure client.
    request_timeout_s: float = Field(default=30.0)
