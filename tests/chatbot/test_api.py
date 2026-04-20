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

from unittest.mock import MagicMock

from openai import APIError, APITimeoutError

from tests.chatbot.conftest import _make_tool_call

# ── Shared payload builders ────────────────────────────────────────────────────

def collection_payload(
    messages=None,
    user_info=None,
    user_confirmed: bool = False,
    confirmed_data=None,
):
    payload = {
        "phase": "collection",
        "messages": messages or [],
        "user_info": user_info,
    }
    if user_confirmed:
        payload["user_confirmed"] = True
    if confirmed_data is not None:
        payload["confirmed_data"] = confirmed_data
    return payload


def qa_payload(messages, user_info):
    return {
        "phase": "qa",
        "messages": messages,
        "user_info": user_info,
    }


USER_MESSAGE = [{"role": "user", "content": "שלום, אני רוצה לרשום"}]
ASSISTANT_MESSAGE = [{"role": "assistant", "content": "ברוך הבא!"}]

# A collection-phase history where the user has just confirmed the summary.
# Used by the transition tests because the API now requires an affirmative
# user turn before accepting a submit_user_info tool call.
CONFIRMED_COLLECTION_HISTORY = [
    {"role": "user", "content": "שלום, אני רוצה לרשום"},
    {"role": "assistant", "content": "בבקשה אשר/י שהפרטים נכונים."},
    {"role": "user", "content": "כן, הכול נכון"},
]


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

    def test_retrieval_block_present(self, api_client):
        body = api_client.get("/health").json()
        assert "retrieval" in body
        assert body["retrieval"]["ready"] is True
        assert body["retrieval"]["indexed_chunks"] == 2
        assert body["retrieval"]["top_k"] >= 1
        assert body["retrieval"]["embedding_deployment"]


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
            "/api/v1/chat", json=collection_payload(CONFIRMED_COLLECTION_HISTORY)
        ).json()
        assert body["transition"] is True

    def test_phase_switches_to_qa(self, api_client, transition_reply):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(CONFIRMED_COLLECTION_HISTORY)
        ).json()
        assert body["phase"] == "qa"

    def test_extracted_user_info_populated(self, api_client, transition_reply, sample_user_info):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(CONFIRMED_COLLECTION_HISTORY)
        ).json()
        info = body["extracted_user_info"]
        assert info is not None
        assert info["first_name"] == sample_user_info["first_name"]
        assert info["hmo_name"] == sample_user_info["hmo_name"]
        assert info["insurance_tier"] == sample_user_info["insurance_tier"]

    def test_all_user_info_fields_present(self, api_client, transition_reply):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(CONFIRMED_COLLECTION_HISTORY)
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
            "/api/v1/chat", json=collection_payload(CONFIRMED_COLLECTION_HISTORY)
        ).json()
        assert body["transition"] is True
        assert len(body["message"]) > 0   # fallback text was generated

    def test_invalid_tool_arguments_returns_500(self, api_client, make_llm_response):
        tc = MagicMock()
        tc.function.name = "submit_user_info"
        tc.function.arguments = "{not valid json"   # malformed JSON
        reply = make_llm_response("אישור", tool_call=tc)
        api_client._mock_openai.chat.completions.create.return_value = reply
        resp = api_client.post(
            "/api/v1/chat", json=collection_payload(CONFIRMED_COLLECTION_HISTORY)
        )
        assert resp.status_code == 500


class TestConfirmationGate:
    """The submit_user_info tool call must be gated on an affirmative user turn.

    These tests cover the eager-tool-calling failure mode: even if the LLM
    fires the tool, the API must refuse to transition until the user has
    actually confirmed in the messages array.
    """

    def test_eager_tool_call_without_confirmation_does_not_transition(
        self, api_client, transition_reply
    ):
        # User's last message is a normal greeting, not a confirmation.
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["transition"] is False
        assert body["phase"] == "collection"
        assert body["extracted_user_info"] is None

    def test_negation_in_user_turn_blocks_transition(
        self, api_client, transition_reply
    ):
        history = [
            {"role": "user", "content": "שלום"},
            {"role": "assistant", "content": "אנא אשר/י"},
            {"role": "user", "content": "לא נכון, יש טעות בגיל"},
        ]
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post("/api/v1/chat", json=collection_payload(history)).json()
        assert body["transition"] is False
        assert body["phase"] == "collection"

    def test_english_confirmation_accepted(
        self, api_client, transition_reply
    ):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Please confirm"},
            {"role": "user", "content": "yes, that's right"},
        ]
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post("/api/v1/chat", json=collection_payload(history)).json()
        assert body["transition"] is True
        assert body["phase"] == "qa"

    def test_eager_tool_call_surfaces_pending_user_info(
        self, api_client, transition_reply, sample_user_info
    ):
        """When the gate blocks, the UI still gets pending_user_info so the
        user can click a confirm button instead of retyping 'yes'."""
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["transition"] is False
        assert body["confirmation_pending"] is True
        assert body["pending_user_info"] is not None
        assert body["pending_user_info"]["first_name"] == sample_user_info["first_name"]


# ══════════════════════════════════════════════════════════════════════════════
# Typed confirmation flag (preferred path)
# ══════════════════════════════════════════════════════════════════════════════

class TestTypedConfirmationFlag:
    """UI sets ``user_confirmed=True`` → we trust the action without sniffing text."""

    def test_typed_flag_opens_gate_without_affirmative_text(
        self, api_client, transition_reply
    ):
        # No "yes" in the last user turn — the typed flag alone should suffice.
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Please confirm"},
        ]
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat",
            json=collection_payload(history, user_confirmed=True),
        ).json()
        assert body["transition"] is True
        assert body["phase"] == "qa"

    def test_typed_flag_with_matching_confirmed_data_accepted(
        self, api_client, transition_reply, sample_user_info
    ):
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat",
            json=collection_payload(
                USER_MESSAGE,
                user_confirmed=True,
                confirmed_data=sample_user_info,
            ),
        ).json()
        assert body["transition"] is True
        assert body["extracted_user_info"]["first_name"] == sample_user_info["first_name"]

    def test_typed_flag_with_mismatched_confirmed_data_blocks_transition(
        self, api_client, transition_reply, sample_user_info
    ):
        # If the LLM swaps a field between review and submit, the gate must refuse.
        tampered = dict(sample_user_info)
        tampered["first_name"] = "מישהו אחר"
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat",
            json=collection_payload(
                USER_MESSAGE,
                user_confirmed=True,
                confirmed_data=tampered,
            ),
        ).json()
        assert body["transition"] is False
        assert body["phase"] == "collection"
        assert body["confirmation_pending"] is True

    def test_typed_flag_defaults_to_false(self, api_client, transition_reply):
        """Omitting user_confirmed must behave exactly like user_confirmed=False."""
        api_client._mock_openai.chat.completions.create.return_value = transition_reply
        body = api_client.post(
            "/api/v1/chat",
            json=collection_payload(USER_MESSAGE),  # no user_confirmed sent
        ).json()
        # USER_MESSAGE is just a greeting — text fallback won't open the gate either.
        assert body["transition"] is False


# ══════════════════════════════════════════════════════════════════════════════
# request_user_confirmation tool
# ══════════════════════════════════════════════════════════════════════════════

class TestRequestUserConfirmation:
    """The LLM calls request_user_confirmation to hand control to the UI's
    confirm dialog — phase stays collection, no transition, data surfaced."""

    def test_confirmation_pending_flag_set(
        self, api_client, request_confirmation_reply
    ):
        api_client._mock_openai.chat.completions.create.return_value = (
            request_confirmation_reply
        )
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["confirmation_pending"] is True

    def test_no_phase_transition(self, api_client, request_confirmation_reply):
        api_client._mock_openai.chat.completions.create.return_value = (
            request_confirmation_reply
        )
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        assert body["phase"] == "collection"
        assert body["transition"] is False
        assert body["extracted_user_info"] is None

    def test_pending_user_info_populated(
        self, api_client, request_confirmation_reply, sample_user_info
    ):
        api_client._mock_openai.chat.completions.create.return_value = (
            request_confirmation_reply
        )
        body = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        ).json()
        info = body["pending_user_info"]
        assert info is not None
        assert info["first_name"] == sample_user_info["first_name"]
        assert info["hmo_name"] == sample_user_info["hmo_name"]
        assert info["insurance_tier"] == sample_user_info["insurance_tier"]

    def test_malformed_arguments_returns_500(
        self, api_client, make_llm_response
    ):
        tc = MagicMock()
        tc.function.name = "request_user_confirmation"
        tc.function.arguments = "{broken json"
        api_client._mock_openai.chat.completions.create.return_value = (
            make_llm_response("סיכום", tool_call=tc)
        )
        resp = api_client.post(
            "/api/v1/chat", json=collection_payload(USER_MESSAGE)
        )
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

    def test_retriever_query_carries_conversation_context(
        self, api_client, qa_reply, sample_user_info
    ):
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        api_client.post(
            "/api/v1/chat",
            json=qa_payload(
                [
                    {"role": "user", "content": "מה הכיסוי לטיפולי שיניים?"},
                    {"role": "assistant", "content": "יש בדיקות חינם פעמיים בשנה."},
                    {"role": "user", "content": "האם זה חל גם על הילדים שלי?"},
                ],
                sample_user_info,
            ),
        )
        # The retriever query should contain BOTH the latest user message
        # AND the prior context so follow-up pronouns still retrieve
        # relevant chunks (conversation-aware retrieval).
        api_client._mock_retriever.search.assert_called_once()
        args, _ = api_client._mock_retriever.search.call_args
        query = args[0]
        assert "האם זה חל גם על הילדים שלי?" in query
        assert "טיפולי שיניים" in query or "בדיקות חינם" in query

    def test_falls_back_to_full_kb_when_retriever_not_ready(
        self, api_client, qa_reply, sample_user_info
    ):
        """Retriever down → we should still answer, using the full KB."""
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        api_client._mock_retriever.is_ready.return_value = False
        resp = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        )
        assert resp.status_code == 200
        # The LLM must still be called — the whole KB is now the context.
        api_client._mock_openai.chat.completions.create.assert_called_once()

    def test_falls_back_to_full_kb_when_retrieval_returns_nothing(
        self, api_client, qa_reply, sample_user_info
    ):
        """Embedding call returned nothing → fall back rather than fail."""
        api_client._mock_openai.chat.completions.create.return_value = qa_reply
        api_client._mock_retriever.search.return_value = []
        resp = api_client.post(
            "/api/v1/chat",
            json=qa_payload(USER_MESSAGE, sample_user_info),
        )
        assert resp.status_code == 200
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
