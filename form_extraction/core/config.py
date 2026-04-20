"""Runtime configuration for the form-extraction pipeline.

Extends the shared `AzureOpenAISettings` with Document Intelligence
credentials; everything else (Azure OpenAI + env-file loading) is inherited.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr

from common.config import AzureOpenAISettings


class Settings(AzureOpenAISettings):
    azure_doc_intelligence_endpoint: str = Field(default="")
    azure_doc_intelligence_key: SecretStr = Field(default=SecretStr(""))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
