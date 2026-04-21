"""Basic FastAPI endpoint tests. All Azure calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

from openai import APITimeoutError

USER_MESSAGE = [{"role": "user", "content": "שלום, אני רוצה לרשום"}]


def _collection_payload(messages):
    return {"phase": "collection", "messages": messages, "user_info": None}


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
    assert body["message"]


def test_submit_user_info_triggers_phase_transition(
    api_client, transition_reply, sample_user_info
):
    api_client._mock_openai.chat.completions.create.return_value = transition_reply
    body = api_client.post("/api/v1/chat", json=_collection_payload(USER_MESSAGE)).json()
    assert body["transition"] is True
    assert body["phase"] == "qa"
    assert body["extracted_user_info"]["first_name"] == sample_user_info["first_name"]


def test_qa_phase_returns_answer(api_client, qa_reply, sample_user_info):
    api_client._mock_openai.chat.completions.create.return_value = qa_reply
    body = api_client.post("/api/v1/chat", json=_qa_payload(USER_MESSAGE, sample_user_info)).json()
    assert body["phase"] == "qa"
    assert body["message"]


def test_llm_timeout_returns_504(api_client):
    api_client._mock_openai.chat.completions.create.side_effect = (
        APITimeoutError(request=MagicMock())
    )
    resp = api_client.post("/api/v1/chat", json=_collection_payload(USER_MESSAGE))
    assert resp.status_code == 504
