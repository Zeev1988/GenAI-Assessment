"""Shared pytest fixtures for the chatbot test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from chatbot.core.knowledge import KnowledgeBase
from chatbot.core.retrieval import Retriever


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_user_info() -> dict[str, Any]:
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


def _build_response(content: str | None, tool_call: MagicMock | None = None) -> MagicMock:
    """Mock an openai.types.chat.ChatCompletion response."""
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = [tool_call] if tool_call else None
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_tool_call(args: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = "submit_user_info"
    tc.function.arguments = json.dumps(args)
    return tc


@pytest.fixture
def make_llm_response():
    return _build_response


@pytest.fixture
def mock_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb._content = {
        "dentel_services": (
            "<h2>מרפאות שיניים</h2>"
            "<table><tr><th>שירות</th><th>מכבי</th></tr>"
            "<tr><td>ניקוי שיניים</td><td><strong>זהב:</strong> חינם פעמיים בשנה</td></tr>"
            "</table>"
        ),
        "optometry_services": (
            "<h2>אופטומטריה</h2>"
            "<table><tr><th>שירות</th><th>מכבי</th></tr>"
            "<tr><td>בדיקת ראייה</td><td><strong>זהב:</strong> חינם פעם בשנה</td></tr>"
            "</table>"
        ),
    }
    kb._loaded = True
    return kb


@pytest.fixture
def mock_retriever(mock_kb) -> MagicMock:
    chunks = mock_kb.chunks()[:2]
    retriever = MagicMock(spec=Retriever)
    retriever.is_ready.return_value = True
    retriever.chunk_count.return_value = len(chunks)
    retriever.search = AsyncMock(
        return_value=[(c, 0.9 - 0.1 * i) for i, c in enumerate(chunks)]
    )
    retriever.index = AsyncMock(return_value=None)
    return retriever


@pytest.fixture
def api_client(mock_kb, mock_retriever) -> TestClient:
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock()
    mock_openai.embeddings.create = AsyncMock()

    from chatbot.api.main import app, get_client, get_retriever

    app.dependency_overrides[get_client] = lambda: mock_openai
    app.dependency_overrides[get_retriever] = lambda: mock_retriever

    with patch("chatbot.api.main.get_knowledge_base", return_value=mock_kb):
        client = TestClient(app, raise_server_exceptions=False)
        client._mock_openai = mock_openai
        client._mock_retriever = mock_retriever
        try:
            yield client
        finally:
            app.dependency_overrides.clear()


@pytest.fixture
def collection_reply(make_llm_response):
    return make_llm_response("מה שמך המלא?")


@pytest.fixture
def transition_reply(make_llm_response, sample_user_info):
    tc = _make_tool_call(sample_user_info)
    return make_llm_response("תודה! כל הפרטים נשמרו.", tool_call=tc)


@pytest.fixture
def qa_reply(make_llm_response):
    return make_llm_response("במסלול זהב של מכבי ניקוי שיניים הוא חינם פעמיים בשנה.")
