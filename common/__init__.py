"""Utilities shared by both Part 1 (form_extraction) and Part 2 (chatbot).

Keeping the shared surface tiny on purpose: one Azure-settings base class
and one logger factory.  Anything larger belongs in the package that owns
the specific concern.
"""

from common.config import AzureOpenAISettings, ENV_FILE, PROJECT_ROOT
from common.logger import get_logger

__all__ = [
    "AzureOpenAISettings",
    "ENV_FILE",
    "PROJECT_ROOT",
    "get_logger",
]
