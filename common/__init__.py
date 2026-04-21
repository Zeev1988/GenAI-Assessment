"""Utilities shared by Part 1 (form_extraction) and Part 2 (chatbot)."""

from common.config import ENV_FILE, PROJECT_ROOT, AzureOpenAISettings
from common.logger import get_logger

__all__ = [
    "AzureOpenAISettings",
    "ENV_FILE",
    "PROJECT_ROOT",
    "get_logger",
]
