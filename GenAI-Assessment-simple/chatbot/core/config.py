"""Runtime configuration for the chatbot service.

Settings are loaded from the project-root .env file (same file used by Part 1)
so a single credentials file covers the whole repository.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root:
# chatbot/core/config.py → chatbot/core → chatbot → project root
_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _ROOT / ".env"


def _find_knowledge_base() -> Path:
    """Locate the phase2_data directory by walking up from this file.

    This is robust to the package being imported from any depth in the tree
    (e.g. after pytest modifies sys.path, or if chatbot/ lives at a different
    nesting level than expected).  We prefer an existing directory over a
    hardcoded relative path.
    """
    target = Path("tests") / "test_data" / "phase2_data"
    current = Path(__file__).resolve().parent
    for _ in range(6):          # search up to 6 levels up
        candidate = current / target
        if candidate.exists():
            return candidate
        current = current.parent
        print("2132132")
    # Nothing found — fall back to parents[2] so the error message at least
    # shows a meaningful path.
    return Path(__file__).resolve().parents[2] / target


_KNOWLEDGE_BASE_DEFAULT = _find_knowledge_base()


class ChatBotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure OpenAI ───────────────────────────────────────────────────────────
    azure_openai_endpoint: str = Field(default="")
    azure_openai_key: SecretStr = Field(default=SecretStr(""))
    azure_openai_api_version: str = Field(default="2024-12-01-preview")
    azure_openai_deployment: str = Field(default="gpt-4o")
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