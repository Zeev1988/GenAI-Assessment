"""
Shared pytest fixtures for the chatbot test suite.

Fixture hierarchy
-----------------
test_data_path          Path to the real HTML knowledge-base files
sample_user_info          Minimal valid member dict used across tests
make_llm_response         Factory that builds mock OpenAI chat-completion responses
mock_kb                   A pre-loaded KnowledgeBase instance with synthetic content
mock_retriever            A pre-indexed Retriever with stub search() results
api_client                FastAPI TestClient with OpenAI + KB + retriever mocked
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from chatbot.core.knowledge import KnowledgeBase
from chatbot.core.retrieval import Retriever

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


# ── Retriever mock ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_retriever(mock_kb) -> MagicMock:
    """A Retriever spec-mock that returns the first two mock_kb chunks on search.

    Short-circuits the embedding API so Q&A tests never hit Azure.  Individual
    tests can override ``mock_retriever.search.return_value`` to tune ranking,
    or ``mock_retriever.is_ready.return_value = False`` to simulate an
    unready index.

    Notes on async:  ``Retriever.search`` is ``async def`` in production, so
    we expose it as an ``AsyncMock``.  ``is_ready`` / ``chunk_count`` stay
    synchronous.
    """
    chunks = mock_kb.chunks()[:2]
    retriever = MagicMock(spec=Retriever)
    retriever.is_ready.return_value = True
    retriever.chunk_count.return_value = len(chunks)
    retriever.search = AsyncMock(
        return_value=[(c, 0.9 - 0.1 * i) for i, c in enumerate(chunks)]
    )
    retriever.index = AsyncMock(return_value=None)
    return retriever


# ── FastAPI TestClient ─────────────────────────────────────────────────────────

@pytest.fixture
def api_client(mock_kb, mock_retriever) -> TestClient:
    """
    TestClient for the FastAPI app.

    The Azure OpenAI client, knowledge base, and retriever are all mocked so
    the tests never hit external services.  Individual tests override
    ``_mock_openai.chat.completions.create`` (an ``AsyncMock``) to return
    specific responses — setting ``return_value`` is enough because
    ``AsyncMock`` returns it from the awaitable.

    The FastAPI client and retriever are injected via the ``get_client`` /
    ``get_retriever`` Depends providers rather than by patching a global;
    tests override them using ``app.dependency_overrides``.
    """
    mock_openai = MagicMock()
    # The production code uses ``await client.chat.completions.create(...)``
    # and ``await client.embeddings.create(...)`` — those must return
    # awaitables, so we upgrade just those methods to ``AsyncMock`` (the
    # rest of the client can stay a plain ``MagicMock``).
    mock_openai.chat.completions.create = AsyncMock()
    mock_openai.embeddings.create = AsyncMock()

    from chatbot.api.main import app, get_client, get_retriever

    app.dependency_overrides[get_client] = lambda: mock_openai
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    with patch("chatbot.api.main.get_knowledge_base", return_value=mock_kb):
        client = TestClient(app, raise_server_exceptions=False)
        # Attach the mocks so individual tests can configure them.
        client._mock_openai = mock_openai
        client._mock_retriever = mock_retriever
        try:
            yield client
        finally:
            app.dependency_overrides.clear()


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
