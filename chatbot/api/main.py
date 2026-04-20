"""
FastAPI microservice — Israeli HMO Chatbot API
==============================================

Design principles
-----------------
• **Fully stateless**: every request carries the complete conversation history
  and user info; the server stores nothing between calls.
• **Two-phase flow**:
    - ``collection`` — LLM gathers member details via natural dialogue and
      signals completion by calling the ``submit_user_info`` tool.
    - ``qa``         — LLM answers health-fund questions using the HTML
      knowledge base and the member's profile.
• **Concurrent users**: FastAPI's async request handling + a single shared
  (thread-safe) Azure OpenAI client supports many simultaneous sessions.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIError, APITimeoutError, AzureOpenAI
from pydantic import BaseModel, Field, field_validator

from common import get_logger

from ..core.config import get_settings
from ..core.knowledge import get_knowledge_base
from ..core.prompts import (
    COLLECTION_SYSTEM_PROMPT,
    REQUEST_USER_CONFIRMATION_TOOL,
    SUBMIT_USER_INFO_TOOL,
    build_qa_system_prompt,
)
from ..core.retrieval import Retriever

settings = get_settings()
logger = get_logger(
    __name__,
    level=settings.log_level,
    log_file=settings.log_file,
)


# ── Pydantic request / response models ────────────────────────────────────────

class Message(BaseModel):
    """A single turn in the conversation history."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Payload sent by the Streamlit frontend on every user turn."""

    phase: str = Field(..., pattern="^(collection|qa)$")
    messages: list[Message] = Field(
        default_factory=list,
        description=(
            "Full conversation history so far, including the latest user message. "
            "May be empty on the very first turn (server returns the opening greeting)."
        ),
    )
    user_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="Confirmed member data — required when phase='qa'.",
    )
    # Typed confirmation channel (preferred over text sniffing).
    #
    # Set by the UI *only* in response to an explicit user action — e.g., the
    # user clicking "Confirm" on the dialog the frontend rendered after the
    # LLM called ``request_user_confirmation``.  When True, the backend will
    # accept a subsequent ``submit_user_info`` tool call without having to
    # classify the user's free-text reply.
    user_confirmed: bool = Field(
        default=False,
        description=(
            "True when the UI is relaying an explicit user confirmation "
            "(e.g., a confirm-button click).  Legacy clients can omit this "
            "and rely on the text-based fallback gate."
        ),
    )
    confirmed_data: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Snapshot of the data the user just confirmed, echoed by the UI "
            "so the server can cross-check it against the LLM's tool-call "
            "arguments.  Ignored unless user_confirmed=True."
        ),
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Client-generated UUID for end-to-end tracing.",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        # Allow empty list only for the collection phase (first-load greeting).
        return v


class ChatResponse(BaseModel):
    """Payload returned to the Streamlit frontend."""

    message: str = Field(description="Assistant reply to display in the chat.")
    phase: str = Field(description="Active phase after this response.")
    transition: bool = Field(
        default=False,
        description="True when the collection phase just completed.",
    )
    extracted_user_info: Optional[dict[str, Any]] = Field(
        default=None,
        description="Parsed member data (only set when transition=True).",
    )
    confirmation_pending: bool = Field(
        default=False,
        description=(
            "True when the LLM has called request_user_confirmation and the "
            "UI should render an explicit confirm/cancel dialog.  The data "
            "to display lives in pending_user_info."
        ),
    )
    pending_user_info: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Data the user is being asked to confirm.  Set only when "
            "confirmation_pending=True.  The UI should round-trip this back "
            "in the confirmed_data field on the next request."
        ),
    )
    request_id: str
    processing_time_ms: int = Field(default=0)


# ── Application lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the knowledge base and build the ADA-002 retrieval index."""
    logger.info("=== HMO Chatbot API starting up ===")
    kb = get_knowledge_base()
    if not kb.is_loaded():
        logger.error("Knowledge base NOT loaded — Q&A will not work")
    else:
        logger.info("Knowledge base ready: %d topics loaded", kb.topic_count())
        _get_retriever().index(kb.chunks())
    yield
    logger.info("=== HMO Chatbot API shutting down ===")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="HMO Chatbot API",
    description=(
        "Stateless microservice for answering Israeli health-fund (קופות חולים) "
        "service questions after collecting member information via LLM-driven dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to known origins in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Azure OpenAI client ────────────────────────────────────────────────────────

_client: Optional[AzureOpenAI] = None
_retriever: Optional[Retriever] = None


def _get_client() -> AzureOpenAI:
    """Return the singleton Azure OpenAI client, creating it on first call."""
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            timeout=settings.request_timeout_s,
            # The SDK retries with exponential back-off on 429 / 503 /
            # connection errors.  3 attempts strikes a balance between
            # resilience and not hammering a rate-limited endpoint.
            # Note: APITimeoutError is intentionally NOT retried by the SDK
            # (we surface it as a 504 so the client can decide).
            max_retries=3,
        )
        logger.info(
            "Azure OpenAI client initialised (deployment=%s, api_version=%s)",
            settings.azure_openai_deployment,
            settings.azure_openai_api_version,
        )
    return _client


def _get_retriever() -> Retriever:
    """Return the singleton Retriever, creating it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            client=_get_client(),
            embedding_deployment=settings.azure_openai_embedding_deployment,
        )
    return _retriever


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _openai_messages(
    system_prompt: str,
    history: list[Message],
) -> list[dict]:
    """Build the messages list expected by the OpenAI chat completions API."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    msgs += [{"role": m.role, "content": m.content} for m in history]
    return msgs


def _call_llm(
    messages: list[dict],
    request_id: str,
    tools: Optional[list[dict]] = None,
) -> Any:
    """Call Azure OpenAI and return the raw response, with structured error handling."""
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": settings.azure_openai_deployment,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        logger.debug(
            "[%s] Calling LLM — %d messages, tools=%s",
            request_id,
            len(messages),
            bool(tools),
        )
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        if usage:
            logger.info(
                "[%s] LLM usage — prompt=%d, completion=%d, total=%d tokens",
                request_id,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        return response
    except APITimeoutError as exc:
        logger.error("[%s] LLM timeout: %s", request_id, exc)
        raise HTTPException(
            status_code=504,
            detail="The language model timed out. Please try again.",
        )
    except APIConnectionError as exc:
        logger.error("[%s] LLM connection error: %s", request_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Cannot reach the language model service. Please try again.",
        )
    except APIError as exc:
        # status_code lives on APIStatusError (a subclass), not on APIError itself.
        # message is always present on the base class, but guard with getattr to
        # be safe across library versions.
        status = getattr(exc, "status_code", "N/A")
        message = getattr(exc, "message", str(exc))
        logger.error("[%s] LLM API error (status=%s): %s", request_id, status, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Language model error: {message}",
        )


# ── Phase handlers ─────────────────────────────────────────────────────────────

# ── Confirmation-gate helpers ─────────────────────────────────────────────────
#
# The assignment mandates a stateless microservice — all session state lives
# client-side.  So we can't store a "user confirmed" flag on the server.
#
# Two confirmation channels are supported, in priority order:
#
#   1. Typed payload flag (preferred).  ``ChatRequest.user_confirmed=True``
#      means the UI is relaying an explicit, unambiguous user action — e.g.
#      a "Confirm" button click on the dialog rendered after the LLM called
#      ``request_user_confirmation``.  No text classification is needed.
#
#   2. Text-based fallback (legacy).  If the UI doesn't set the typed flag
#      (older client, or first call before the confirm dialog exists), we
#      fall back to inspecting the latest user turn in the messages array
#      for an affirmative phrase.  Lower precision — but a backstop so the
#      tool-call gate never opens purely because the LLM fired the tool.
#
# Either way the check operates purely on the payload the client sends us,
# so architectural statelessness is preserved.

# Affirmative phrases recognised by the text-based fallback gate.  We
# deliberately keep the list short and high-precision: a false negative
# (user re-confirms) is cheaper than a false positive (tool call fires
# before the user agreed).
_CONFIRM_TOKENS_HE: frozenset[str] = frozenset({
    "כן", "אישור", "מאשר", "מאשרת", "נכון", "בסדר", "אוקיי", "אוקי",
    "מסכים", "מסכימה", "מאשר/ת", "הכל נכון", "הכול נכון", "ללא שינויים",
})
_CONFIRM_TOKENS_EN: frozenset[str] = frozenset({
    "yes", "yep", "yeah", "correct", "confirm", "confirmed", "confirming",
    "that's right", "thats right", "all correct", "looks good", "lgtm",
    "approved", "approve", "go ahead", "ok", "okay",
})
# Words that negate a confirmation — if any appear in the same turn, we
# assume the user is correcting something and refuse the tool call.
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "לא", "תקן", "לתקן", "שנה", "שני", "שינוי", "שגוי", "טעות", "לא נכון",
    "no", "not", "wrong", "incorrect", "change", "correct ", "fix", "edit",
    "update", "actually",
})

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _is_affirmative(text: str) -> bool:
    """Return True iff *text* reads as an unambiguous user confirmation."""
    normalised = (text or "").strip().lower()
    if not normalised:
        return False
    tokens = _WORD_RE.findall(normalised)
    if not tokens:
        return False
    token_set = set(tokens)
    # Check phrase-level matches first so multi-word confirmations/negations
    # ("that's right", "לא נכון") aren't tripped up by the tokeniser.
    for phrase in _NEGATION_TOKENS:
        if " " in phrase and phrase in normalised:
            return False
    for phrase in _CONFIRM_TOKENS_HE | _CONFIRM_TOKENS_EN:
        if " " in phrase and phrase in normalised:
            # Still bail out if a negation token is also present.
            return not any(neg in token_set for neg in _NEGATION_TOKENS if " " not in neg)

    if any(neg in token_set for neg in _NEGATION_TOKENS if " " not in neg):
        return False
    return bool(token_set & (_CONFIRM_TOKENS_HE | _CONFIRM_TOKENS_EN))


def _confirmation_present(messages: list[Message]) -> bool:
    """True iff the latest user turn in *messages* reads as a confirmation.

    We look at the message history the client sent us and check the most
    recent ``user`` turn.  No server-side state is consulted.
    """
    for m in reversed(messages):
        if m.role == "user":
            return _is_affirmative(m.content)
    return False


def _parse_tool_arguments(
    tc: Any, request_id: str, tool_label: str,
) -> dict[str, Any]:
    """Parse a tool call's JSON arguments, surfacing a 500 on malformed input."""
    try:
        return json.loads(tc.function.arguments)
    except json.JSONDecodeError as exc:
        logger.error(
            "[%s] Cannot parse %s arguments: %s", request_id, tool_label, exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse {tool_label} arguments.",
        )


def _confirmation_gate_passes(
    req_user_confirmed: bool,
    confirmed_data: Optional[dict[str, Any]],
    tool_args: dict[str, Any],
    messages: list[Message],
    request_id: str,
) -> tuple[bool, str]:
    """Decide whether a submit_user_info tool call should be accepted.

    Returns (passed, reason).  ``reason`` is a short label used for logging
    so we can tell at-a-glance *why* the gate opened (or didn't).
    """
    # 1) Typed payload flag — preferred path.  When the UI sets this, the
    #    user just clicked a confirm button and we trust the action.  We
    #    additionally cross-check ``confirmed_data`` (if provided) against
    #    the tool arguments so the LLM can't quietly swap a field between
    #    the review dialog and the final submit.
    if req_user_confirmed:
        if confirmed_data is not None and confirmed_data != tool_args:
            logger.warning(
                "[%s] user_confirmed=True but confirmed_data != tool args — "
                "refusing submit. diff_keys=%s",
                request_id,
                sorted(set(confirmed_data) ^ set(tool_args))
                or [k for k in tool_args if confirmed_data.get(k) != tool_args.get(k)],
            )
            return False, "data_mismatch"
        return True, "typed_flag"

    # 2) Legacy text-based fallback — only consulted when the typed flag is
    #    absent.  Keeps the old Streamlit client working until it's upgraded
    #    to send ``user_confirmed``.
    if _confirmation_present(messages):
        return True, "text_fallback"

    return False, "no_confirmation"


def _handle_collection(
    messages: list[Message],
    request_id: str,
    user_confirmed: bool = False,
    confirmed_data: Optional[dict[str, Any]] = None,
) -> ChatResponse:
    """
    Drive the information-collection phase.

    The LLM has two tools available:

    • ``request_user_confirmation`` — the correct tool to fire once all
      seven fields have been gathered.  The backend relays the data to the
      UI, which shows an explicit confirm dialog; no phase transition yet.

    • ``submit_user_info`` — finalises registration and transitions to Q&A.
      Gated by the two-channel confirmation check in
      ``_confirmation_gate_passes``: either the typed ``user_confirmed``
      flag (preferred) or the legacy text-based fallback.
    """
    # On the very first load (empty history) inject a hidden 'start' signal
    # so the LLM produces an opening greeting without requiring user input.
    effective_messages = messages
    if not messages:
        effective_messages = [
            Message(role="user", content="[SESSION_START]")
        ]
        logger.info("[%s] First load — injecting SESSION_START", request_id)
    else:
        logger.info(
            "[%s] Collection phase — %d messages in history, user_confirmed=%s",
            request_id,
            len(messages),
            user_confirmed,
        )

    openai_msgs = _openai_messages(COLLECTION_SYSTEM_PROMPT, effective_messages)
    response = _call_llm(
        openai_msgs,
        request_id,
        tools=[REQUEST_USER_CONFIRMATION_TOOL, SUBMIT_USER_INFO_TOOL],
    )

    choice = response.choices[0]
    content: str = choice.message.content or ""
    tool_calls = choice.message.tool_calls

    if tool_calls:
        tc = tool_calls[0]
        tool_name = tc.function.name

        # ── request_user_confirmation → ask UI to render confirm dialog ──
        if tool_name == "request_user_confirmation":
            pending = _parse_tool_arguments(tc, request_id, "request_user_confirmation")
            logger.info(
                "[%s] request_user_confirmation — %s %s | HMO: %s",
                request_id,
                pending.get("first_name"),
                pending.get("last_name"),
                pending.get("hmo_name"),
            )
            # Use the LLM's summary text, or synthesise a minimal one.
            reply = content or (
                "סיכמתי את הפרטים. האם הכול נכון? לחץ/י על 'אישור' כדי להמשיך, "
                "או 'תיקון' כדי לעדכן."
            )
            return ChatResponse(
                message=reply,
                phase="collection",
                confirmation_pending=True,
                pending_user_info=pending,
                request_id=request_id,
            )

        # ── submit_user_info → gated finalisation ────────────────────────
        if tool_name == "submit_user_info":
            user_info = _parse_tool_arguments(tc, request_id, "submit_user_info")

            passed, reason = _confirmation_gate_passes(
                req_user_confirmed=user_confirmed,
                confirmed_data=confirmed_data,
                tool_args=user_info,
                messages=messages,
                request_id=request_id,
            )
            if not passed:
                logger.warning(
                    "[%s] submit_user_info rejected (%s) — asking for confirmation.",
                    request_id,
                    reason,
                )
                reply = content or (
                    "לפני שאני ממשיך — אשמח אם תאשר/י שכל הפרטים שסיכמתי נכונים. "
                    "אם משהו לא מדויק, אפשר לתקן עכשיו."
                )
                # If the LLM jumped straight to submit_user_info without
                # going through request_user_confirmation, surface the data
                # to the UI as a pending confirmation so the user can click
                # a button instead of retyping "yes".
                return ChatResponse(
                    message=reply,
                    phase="collection",
                    confirmation_pending=True,
                    pending_user_info=user_info,
                    request_id=request_id,
                )

            logger.info(
                "[%s] Collection complete (gate=%s) — %s %s | HMO: %s | tier: %s",
                request_id,
                reason,
                user_info.get("first_name"),
                user_info.get("last_name"),
                user_info.get("hmo_name"),
                user_info.get("insurance_tier"),
            )

            # Use the LLM's farewell text, or fall back to a default.
            if not content:
                fn = user_info.get("first_name", "")
                hmo = user_info.get("hmo_name", "")
                tier = user_info.get("insurance_tier", "")
                content = (
                    f"תודה {fn}! כל הפרטים נשמרו בהצלחה. "
                    f"כמבוטח/ת ב{hmo} במסלול {tier}, "
                    "אני כאן לענות על כל שאלה לגבי שירותי הקופה. במה אוכל לעזור לך?"
                )

            return ChatResponse(
                message=content,
                phase="qa",
                transition=True,
                extracted_user_info=user_info,
                request_id=request_id,
            )

        # Unknown tool name — log and fall through to plain dialogue reply.
        logger.warning(
            "[%s] Unknown tool call from LLM: %s", request_id, tool_name,
        )

    # ── Normal dialogue turn ───────────────────────────────────────────────────
    logger.info("[%s] Collection dialogue — reply: %d chars", request_id, len(content))
    return ChatResponse(
        message=content,
        phase="collection",
        request_id=request_id,
    )


# Maximum number of user turns (plus the most recent assistant turn) to blend
# into the retrieval query.  Small enough to keep the embedding input focused,
# large enough to carry pronoun / follow-up context across a turn or two.
_RETRIEVAL_HISTORY_TURNS = 3


def _build_retrieval_query(messages: list[Message]) -> str:
    """Build a context-aware retrieval query from the conversation history.

    Concatenates the last few user turns and the most recent assistant turn
    so follow-ups like "does that apply to children?" still carry the topic
    ("dental care") into the embedding.  Returns "" when there is no user
    content at all.
    """
    if not messages:
        return ""

    # Most recent user turn is the anchor — it's always included.
    user_turns = [m.content for m in messages if m.role == "user"]
    if not user_turns:
        return ""
    recent_users = user_turns[-_RETRIEVAL_HISTORY_TURNS:]

    # Attach the most recent assistant reply (if any) so pronouns like "that"
    # / "it" / "הוא" have their referent in the embedding input.
    last_assistant = next(
        (m.content for m in reversed(messages) if m.role == "assistant"),
        "",
    )

    parts = recent_users[:]
    if last_assistant:
        # Insert assistant context *between* the earlier user turns and the
        # latest one so the latest query dominates the embedding.
        parts.insert(-1, last_assistant)

    return "\n".join(p.strip() for p in parts if p and p.strip())


def _full_kb_chunks(request_id: str) -> list[Any]:
    """Return every Chunk in the knowledge base, for the retrieval fallback.

    Used when the embedding index isn't ready — we stuff the whole (small)
    corpus into the prompt so Q&A still works rather than crashing with 503.
    """
    kb = get_knowledge_base()
    if not kb.is_loaded():
        return []
    chunks = kb.chunks()
    logger.warning(
        "[%s] Retrieval fallback — injecting full KB (%d chunks) into prompt",
        request_id,
        len(chunks),
    )
    return chunks


def _handle_qa(
    messages: list[Message],
    user_info: dict[str, Any],
    request_id: str,
) -> ChatResponse:
    """Answer health-fund questions using retrieval + the member's profile.

    If the embedding retriever isn't ready at request time, we fall back to
    stuffing the full knowledge base into the prompt (the corpus is small
    enough for GPT-4o's context window).  Only when the KB itself is
    unavailable do we surface a 503.
    """
    retriever = _get_retriever()
    retrieval_query = _build_retrieval_query(messages)

    if retriever.is_ready():
        results = retriever.search(retrieval_query, settings.retrieval_top_k)
        if not results:
            # The index is healthy but the query is empty or embedding failed.
            # Prefer full-KB fallback over a blank 502: the user still gets
            # an answer and the LLM can tell them if the KB doesn't cover it.
            logger.warning(
                "[%s] Retrieval returned no results for query=%r — falling back to full KB",
                request_id,
                retrieval_query[:80],
            )
            chunks = _full_kb_chunks(request_id)
            if not chunks:
                raise HTTPException(
                    status_code=503,
                    detail="The knowledge base is not loaded. Please try again shortly.",
                )
            mode = "full_kb_fallback"
            scored_chunks: list[tuple[Any, float]] = [(c, 0.0) for c in chunks]
        else:
            mode = "retrieval"
            scored_chunks = list(results)
    else:
        chunks = _full_kb_chunks(request_id)
        if not chunks:
            logger.error("[%s] Retrieval down and KB empty", request_id)
            raise HTTPException(
                status_code=503,
                detail="The knowledge base is not loaded. Please try again shortly.",
            )
        mode = "full_kb_fallback"
        scored_chunks = [(c, 0.0) for c in chunks]

    topics_hit = sorted({chunk.topic for chunk, _ in scored_chunks})
    scores_preview = ", ".join(f"{score:.3f}" for _, score in scored_chunks)
    separator = "\n" + "=" * 80 + "\n"
    knowledge_content = separator.join(chunk.prompt_block for chunk, _ in scored_chunks)

    system_prompt = build_qa_system_prompt(user_info, knowledge_content)
    openai_msgs = _openai_messages(system_prompt, messages)

    logger.info(
        "[%s] Q&A phase — %s | HMO: %s | tier: %s | %d messages | mode=%s "
        "| chunks=%d/%d | topics=%s | scores=[%s]",
        request_id,
        user_info.get("first_name"),
        user_info.get("hmo_name"),
        user_info.get("insurance_tier"),
        len(messages),
        mode,
        len(scored_chunks),
        retriever.chunk_count(),
        topics_hit,
        scores_preview,
    )

    response = _call_llm(openai_msgs, request_id)
    answer: str = response.choices[0].message.content or ""

    logger.info("[%s] Q&A answer generated (%d chars, mode=%s)", request_id, len(answer), mode)
    return ChatResponse(
        message=answer,
        phase="qa",
        request_id=request_id,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health_check():
    """Returns service status, knowledge-base readiness, and retriever status."""
    kb = get_knowledge_base()
    retriever = _get_retriever()
    return {
        "status": "ok",
        "knowledge_base_loaded": kb.is_loaded(),
        "topic_count": kb.topic_count(),
        "topics": kb.topic_titles(),
        "deployment": settings.azure_openai_deployment,
        "retrieval": {
            "ready": retriever.is_ready(),
            "indexed_chunks": retriever.chunk_count(),
            "top_k": settings.retrieval_top_k,
            "embedding_deployment": settings.azure_openai_embedding_deployment,
        },
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    description=(
        "Stateless endpoint: the client sends the full conversation history "
        "with each request.  Returns the assistant reply and any phase transition."
    ),
)
async def chat(req: ChatRequest):
    start = time.perf_counter()
    logger.info(
        "POST /api/v1/chat | request_id=%s | phase=%s | msgs=%d",
        req.request_id,
        req.phase,
        len(req.messages),
    )

    if req.phase == "collection":
        result = _handle_collection(
            req.messages,
            req.request_id,
            user_confirmed=req.user_confirmed,
            confirmed_data=req.confirmed_data,
        )
    else:
        # Q&A requires user_info to be present.
        if not req.user_info:
            logger.warning(
                "[%s] Q&A request missing user_info", req.request_id
            )
            raise HTTPException(
                status_code=422,
                detail="user_info is required for the Q&A phase.",
            )
        result = _handle_qa(req.messages, req.user_info, req.request_id)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result.processing_time_ms = elapsed_ms

    logger.info(
        "[%s] Response sent in %dms | phase=%s | transition=%s",
        req.request_id,
        elapsed_ms,
        result.phase,
        result.transition,
    )
    return result


# ── Global exception handler ───────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "chatbot.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
