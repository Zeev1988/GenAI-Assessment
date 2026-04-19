"""
FastAPI endpoint tests for chatbot/api/main.py

All Azure OpenAI calls and knowledge-base I/O are mocked — tests run
fully offline.  The test client exercises the full HTTP stack (routing,
middleware, Pydantic validation, error handlers).

Test groups
-----------
TestHealth              GET /health
TestCollectionPhase     POST /api/v1/chat — collection dialogue turns
TestPhaseTransition     POST /api/v1/chat — tool-call → phase switch
TestQAPhase             POST /api/v1/chat — Q&A answers
TestValidation          Pydantic / domain validation (bad payloads → 4xx)
TestErrorHandling       LLM failures surfaced as correct HTTP codes
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import APIError, APITimeoutError

from chatbot.tests.conftest import _make_tool_call


# ── Shared payload builders ────────────────────────────────────────────────────

def collection_payload(messages=None, user_info=None):
    return {
        "phase": "collection",
        "messages": messages or [],
        "user_info": user_info,
    }


def qa_payload(messages, user_info):
    return {
        "phase": "qa",
        "messages": messages,
        "user_info": user_info,
    }


USER_MESSAGE = [{"role": "user", "content": "שלום, אני רוצה לרשום"}]
ASSISTANT_MESSAGE = [{"role": "assistant", "content": "ברוך הבא!"}]


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_200(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_body_has_status_ok(self, api_client):
        body = api_client.get("/health").json()
        assert body["status"] == "ok"

    def test_knowledge_base_loaded_true(self, api_client):
        body = api_client.get("/health").json()
        assert body["knowledge_base_loaded"] is True

    def test_topic_count_matches_mock(self, api_client):
        # mock_kb fixture has 2 synthetic topics (see conftest.py)
        body = api_client.get("/health").json()
        assert body["topic_count"] == 2

    def test_topics_is_list(self, api_client):
        body = api_client.get("/health").json()
        assert isinstance(body["topics"], list)


# ══════════════════════════════════════════════════════════════════════════════
# Collection phase — normal dialogue turns
# ══════════════════════════════════════════════════════════════════════════════

class TestCollectionPhase:
    def test_returns_200(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        resp = api_client.post("/api/v1/chat", json=collection_payload(USER_MESSAGE))
        assert resp.status_code == 200

    def test_phase_remains_collection(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["phase"] == "collection"

    def test_transition_is_false(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["transition"] is False

    def test_message_is_string(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert isinstance(body["message"], str)
        assert len(body["message"]) > 0

    def test_extracted_user_info_is_null(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["extracted_user_info"] is None

    def test_processing_time_ms_present(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert isinstance(body["processing_time_ms"], int)
        assert body["processing_time_ms"] >= 0

    def test_request_id_echoed(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        payload = collection_payload(USER_MESSAGE)
        payload["request_id"] = "test-req-123"
        body = api_client.post("/api/v1/chat", json=payload).json()
        assert body["request_id"] == "test-req-123"

    def test_empty_messages_list_accepted(self, api_client, collection_reply):
        """Empty messages triggers the opening-greeting path (SESSION_START)."""
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        resp = api_client.post("/api/v1/chat", json=collection_payload([]))
        assert resp.status_code == 200

    def test_multi_turn_history_accepted(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.return_value = collection_reply
        messages = [
            {"role": "user", "content": "שלום"},
            {"role": "assistant", "content": "מה שמך?"},
            {"role": "user", "content": "ישראל"},
        ]
        resp = api_client.post("/api/v1/chat", json=collection_payload(messages))
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# Phase transition (collection → QA)
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseTransition:
    def test_transition_flag_is_true(self, api_client, transition_reply):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["transition"] is True

    def test_phase_switches_to_qa(self, api_client, transition_reply):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["phase"] == "qa"

    def test_extracted_user_info_populated(self, api_client, transition_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        info = body["extracted_user_info"]
        assert info is not None
        assert info["first_name"] == sample_user_info["first_name"]
        assert info["hmo_name"] == sample_user_info["hmo_name"]
        assert info["insurance_tier"] == sample_user_info["insurance_tier"]

    def test_all_user_info_fields_present(self, api_client, transition_reply):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        info = body["extracted_user_info"]
        required_fields = {
            "first_name", "last_name", "id_number", "gender",
            "age", "hmo_name", "hmo_card_number", "insurance_tier",
        }
        assert required_fields.issubset(set(info.keys()))

    def test_fallback_message_when_content_is_none(self, api_client, sample_user_info, make_llm_response):
        """When the LLM sends a tool call with no text content, we generate a default."""
        tc = _make_tool_call(sample_user_info)
        reply = make_llm_response(None, tool_call=tc)  # content=None
        api_client._mock_openai.chat.completions.create.return_value = reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["transition"] is True
        assert len(body["message"]) > 0   # fallback text was generated

    def test_invalid_tool_arguments_returns_500(self, api_client, make_llm_response):
        tc = MagicMock()
        tc.function.name = "submit_user_info"
        tc.function.arguments = "{not valid json"   # malformed JSON
        reply = make_llm_response("אישור", tool_call=tc)
        api_client._mock_openai.chat.completions.create.return_value = reply
        resp = api_client.post("/api/v1/chat", json=collection_payload(USER_MESSAGE))
        assert resp.status_code == 500


# ══════════════════════════════════════════════════════════════════════════════
# Q&A phase
# ══════════════════════════════════════════════════════════════════════════════

class TestQAPhase:
    def test_returns_200(self, api_client, qa_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        resp = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        )
        assert resp.status_code == 200

    def test_phase_remains_qa(self, api_client, qa_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        body = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        ).json()
        assert body["phase"] == "qa"

    def test_transition_is_false(self, api_client, qa_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        body = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        ).json()
        assert body["transition"] is False

    def test_answer_is_non_empty_string(self, api_client, qa_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        body = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        ).json()
        assert isinstance(body["message"], str)
        assert len(body["message"]) > 0

    def test_answer_matches_mock_content(self, api_client, qa_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        body = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        ).json()
        assert "זהב" in body["message"]   # from the qa_reply fixture text

    def test_llm_called_once_per_request(self, api_client, qa_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        api_client._mock_openai.chat.completions.create.reset_mock()
        api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        )
        api_client._mock_openai.chat.completions.create.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Input validation
# ══════════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_invalid_phase_returns_422(self, api_client):
        resp = api_client.post(
            "/api/v1/chat",
            json={"phase": "unknown", "messages": []},
        )
        assert resp.status_code == 422

    def test_missing_phase_returns_422(self, api_client):
        resp = api_client.post("/api/v1/chat", json={"messages": []})
        assert resp.status_code == 422

    def test_invalid_role_in_messages_returns_422(self, api_client):
        resp = api_client.post(
            "/api/v1/chat",
            json={
                "phase": "collection",
                "messages": [{"role": "alien", "content": "hello"}],
            },
        )
        assert resp.status_code == 422

    def test_qa_phase_without_user_info_returns_422(self, api_client, qa_reply):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        resp = api_client.post(
            "/api/v1/chat",
            json={"phase": "qa", "messages": USER_MESSAGE, "user_info": None},
        )
        assert resp.status_code == 422

    def test_non_json_body_returns_422(self, api_client):
        resp = api_client.post(
            "/api/v1/chat",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# LLM error handling
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def _make_api_error(self) -> APIError:
        """Build a minimal APIError for testing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {}
        return APIError("LLM service down", mock_resp, body=None)

    def test_timeout_returns_504(self, api_client, collection_reply):
        api_client._mock_openai.chat.completions.create.side_effect = (
            APITimeoutError(request=MagicMock())
        )
        resp = api_client.post("/api/v1/chat", json=collection_payload(USER_MESSAGE))
        assert resp.status_code == 504

    def test_timeout_error_message_is_user_friendly(self, api_client):
        api_client._mock_openai.chat.completions.create.side_effect = (
            APITimeoutError(request=MagicMock())
        )
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert "detail" in body
        assert len(body["detail"]) > 0

    def test_api_error_returns_502(self, api_client):
        api_client._mock_openai.chat.completions.create.side_effect = (
            self._make_api_error()
        )
        resp = api_client.post("/api/v1/chat", json=collection_payload(USER_MESSAGE))
        assert resp.status_code == 502

    def test_unexpected_exception_returns_500(self, api_client):
        api_client._mock_openai.chat.completions.create.side_effect = (
            RuntimeError("Unexpected crash")
        )
        resp = api_client.post("/api/v1/chat", json=collection_payload(USER_MESSAGE))
        assert resp.status_code == 500

    def test_error_responses_have_detail_field(self, api_client):
        api_client._mock_openai.chat.completions.create.side_effect = (
            APITimeoutError(request=MagicMock())
        )
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert "detail" in body
