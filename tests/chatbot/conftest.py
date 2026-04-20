"""
Shared pytest fixtures for the chatbot test suite.

Fixture hierarchy
-----------------
test_data_path          Path to the real HTML knowledge-base files
sample_user_info          Minimal valid member dict used across tests
make_llm_response         Factory that builds mock OpenAI chat-completion responses
mock_kb                   A pre-loaded KnowledgeBase instance with synthetic content
api_client                FastAPI TestClient with OpenAI + KB both mocked
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from chatbot.core.knowledge import KnowledgeBase


# ── Paths ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_data_path() -> Path:
    """Absolute path to the real HTML knowledge-base directory."""
    return Path(__file__).parent / "test_data"


# ── Domain fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_user_info() -> dict[str, Any]:
    """A fully-populated, valid member profile."""
    return {
        "first_name": "ישראל",
        "last_name": "ישראלי",
        "id_number": "123456789",
        "gender": "זכר",
        "age": 35,
        "hmo_name": "מכבי",
        "hmo_card_number": "987654321",
        "insurance_tier": "זהב",
    }


# ── OpenAI mock helpers ────────────────────────────────────────────────────────

def _make_usage(prompt: int = 100, completion: int = 50) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    return usage


def _make_tool_call(user_info: dict) -> MagicMock:
    """Build a mock tool_call object for submit_user_info."""
    tc = MagicMock()
    tc.function.name = "submit_user_info"
    tc.function.arguments = json.dumps(user_info)
    return tc


def _build_response(
    content: str | None,
    tool_call: MagicMock | None = None,
) -> MagicMock:
    """Return a mock that mimics openai.types.chat.ChatCompletion."""
    resp = MagicMock()
    resp.usage = _make_usage()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = [tool_call] if tool_call else None
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    return resp


@pytest.fixture
def make_llm_response():
    """Factory fixture: call make_llm_response(content, tool_call=None)."""
    return _build_response


# ── Knowledge-base mock ────────────────────────────────────────────────────────

@pytest.fixture
def mock_kb() -> KnowledgeBase:
    """A KnowledgeBase instance loaded with minimal synthetic HTML content."""
    kb = KnowledgeBase()
    kb._content = {
        "dentel_services": (
            "<h2>מרפאות שיניים</h2>"
            "<table><tr><th>שירות</th><th>מכבי</th></tr>"
            "<tr><td>ניקוי שיניים</td>"
            "<td><strong>זהב:</strong> חינם פעמיים בשנה</td></tr>"
            "</table>"
        ),
        "optometry_services": (
            "<h2>אופטומטריה</h2>"
            "<table><tr><th>שירות</th><th>מכבי</th></tr>"
            "<tr><td>בדיקת ראייה</td>"
            "<td><strong>זהב:</strong> חינם פעם בשנה</td></tr>"
            "</table>"
        ),
    }
    kb._loaded = True
    return kb


# ── FastAPI TestClient ─────────────────────────────────────────────────────────

@pytest.fixture
def api_client(mock_kb) -> TestClient:
    """
    TestClient for the FastAPI app.

    Both the Azure OpenAI client and the knowledge base are mocked so the
    tests never hit external services.  Individual tests override
    ``_mock_openai.chat.completions.create`` to return specific responses.
    """
    mock_openai = MagicMock()

    with (
        patch("chatbot.api.main._get_client", return_value=mock_openai),
        patch("chatbot.api.main.get_knowledge_base", return_value=mock_kb),
    ):
        from chatbot.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        # Attach the mock so individual tests can configure return values.
        client._mock_openai = mock_openai
        yield client


# ── Convenience: pre-canned LLM responses ─────────────────────────────────────

@pytest.fixture
def collection_reply(make_llm_response):
    """A plain collection-dialogue response (no tool call)."""
    return make_llm_response("מה שמך המלא?")


@pytest.fixture
def transition_reply(make_llm_response, sample_user_info):
    """A response where the LLM fires submit_user_info → triggers phase transition."""
    tc = _make_tool_call(sample_user_info)
    return make_llm_response("תודה! כל הפרטים נשמרו.", tool_call=tc)


@pytest.fixture
def qa_reply(make_llm_response):
    """A plain Q&A answer response."""
    return make_llm_response(
        "במסלול זהב של מכבי ניקוי שיניים הוא חינם פעמיים בשנה."
    )
