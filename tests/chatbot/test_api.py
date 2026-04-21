"""
Basic FastAPI endpoint tests for chatbot/api/main.py.

All Azure OpenAI calls and knowledge-base I/O are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from openai import APITimeoutError


USER_MESSAGE = [{"role": "user", "content": "שלום, אני רוצה לרשום"}]
POST_CONFIRMATION_HISTORY = [
    {"role": "user", "content": "שלום, אני רוצה לרשום"},
    {"role": "assistant", "content": "בבקשה אשר/י שהפרטים נכונים."},
]


def _collection_payload(messages, user_confirmed: bool = False):
    payload = {"phase": "collection", "messages": messages, "user_info": None}
    if user_confirmed:
        payload["user_confirmed"] = True
    return payload


def _qa_payload(messages, user_info):
    return {"phase": "qa", "messages": messages, "user_info": user_info}


def test_health_endpoint_returns_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["knowledge_base_loaded"] is True


def test_collection_phase_returns_message(api_client, collection_reply):
    api_client._mock_openai.chat.completions.create.return_value = collection_reply
    resp = api_client.post("/api/v1/chat", json=_collection_payload(USER_MESSAGE))
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "collection"
    assert body["transition"] is False
    assert isinstance(body["message"], str) and len(body["message"]) > 0


def test_typed_confirmation_triggers_phase_transition(
    api_client, transition_reply, sample_user_info
):
    """user_confirmed=True + submit_user_info tool call → move to QA phase."""
    api_client._mock_openai.chat.completions.create.return_value = transition_reply
    body = api_client.post(
        "/api/v1/chat",
        json=_collection_payload(POST_CONFIRMATION_HISTORY, user_confirmed=True),
    ).json()
    assert body["transition"] is True
    assert body["phase"] == "qa"
    assert body["extracted_user_info"]["first_name"] == sample_user_info["first_name"]


def test_tool_call_without_confirmation_does_not_transition(
    api_client, transition_reply
):
    """The gate must refuse to transition until the UI relays user_confirmed=True."""
    api_client._mock_openai.chat.completions.create.return_value = transition_reply
    body = api_client.post(
        "/api/v1/chat", json=_collection_payload(USER_MESSAGE)
    ).json()
    assert body["transition"] is False
    assert body["phase"] == "collection"
    assert body["confirmation_pending"] is True


def test_qa_phase_returns_answer(api_client, qa_reply, sample_user_info):
    api_client._mock_openai.chat.completions.create.return_value = qa_reply
    body = api_client.post(
        "/api/v1/chat", json=_qa_payload(USER_MESSAGE, sample_user_info)
    ).json()
    assert body["phase"] == "qa"
    assert isinstance(body["message"], str) and len(body["message"]) > 0


def test_llm_timeout_returns_504(api_client):
    api_client._mock_openai.chat.completions.create.side_effect = (
        APITimeoutError(request=MagicMock())
    )
    resp = api_client.post("/api/v1/chat", json=_collection_payload(USER_MESSAGE))
    assert resp.status_code == 504
