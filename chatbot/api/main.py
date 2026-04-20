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
• **True async concurrency**: the route is ``async def`` and the LLM / embedding
  calls go through ``AsyncAzureOpenAI`` with ``await``, so a single request
  never blocks the event loop while waiting on the model.  Dozens of sessions
  can be in flight at once on a single uvicorn worker.
• **Dependency injection**: the shared OpenAI client and the Retriever are
  built once in ``lifespan`` and handed to the route via FastAPI's
  ``Depends()``.  Tests override them with ``app.dependency_overrides``.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIError, APITimeoutError, AsyncAzureOpenAI
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
    # Typed confirmation channel — the only path by which the backend will
    # accept a ``submit_user_info`` tool call.  Set by the UI only in
    # response to an explicit user action (e.g., clicking "Confirm" on the
    # dialog the frontend rendered after the LLM called
    # ``request_user_confirmation``).
    user_confirmed: bool = Field(
        default=False,
        description=(
            "True when the UI is relaying an explicit user confirmation "
            "(e.g., a confirm-button click).  Required to open the "
            "submit_user_info gate."
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
    """Build the shared Async OpenAI client, Retriever, and embedding index.

    The client and retriever live on ``app.state`` so they're reachable from
    the ``Depends()`` providers below (see ``get_client`` / ``get_retriever``).
    """
    logger.info("=== HMO Chatbot API starting up ===")

    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
        timeout=settings.request_timeout_s,
        max_retries=3,
    )
    logger.info(
        "AsyncAzureOpenAI client initialised (deployment=%s, api_version=%s)",
        settings.azure_openai_deployment,
        settings.azure_openai_api_version,
    )
    retriever = Retriever(
        client=client,
        embedding_deployment=settings.azure_openai_embedding_deployment,
    )
    app.state.openai_client = client
    app.state.retriever = retriever

    kb = get_knowledge_base()
    if not kb.is_loaded():
        logger.error("Knowledge base NOT loaded — Q&A will not work")
    else:
        logger.info("Knowledge base ready: %d topics loaded", kb.topic_count())
        await retriever.index(kb.chunks())

    try:
        yield
    finally:
        await client.close()
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


# ── Dependency providers ───────────────────────────────────────────────────────
#
# FastAPI resolves these per request.  In production they read from the
# singletons stored on ``app.state`` by ``lifespan``.  In tests we override
# them via ``app.dependency_overrides`` to inject mocks.

def get_client(request: Request) -> AsyncAzureOpenAI:
    """Return the process-wide AsyncAzureOpenAI client built in ``lifespan``."""
    client = getattr(request.app.state, "openai_client", None)
    if client is None:  # pragma: no cover — defensive
        raise RuntimeError(
            "OpenAI client not initialised — lifespan did not run."
        )
    return client


def get_retriever(request: Request) -> Retriever:
    """Return the process-wide Retriever built in ``lifespan``."""
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:  # pragma: no cover — defensive
        raise RuntimeError(
            "Retriever not initialised — lifespan did not run."
        )
    return retriever


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _openai_messages(
    system_prompt: str,
    history: list[Message],
) -> list[dict]:
    """Build the messages list expected by the OpenAI chat completions API."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    msgs += [{"role": m.role, "content": m.content} for m in history]
    return msgs


async def _call_llm(
    client: AsyncAzureOpenAI,
    messages: list[dict],
    request_id: str,
    tools: Optional[list[dict]] = None,
) -> Any:
    """Call Azure OpenAI and return the raw response, with structured error handling."""
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
        response = await client.chat.completions.create(**kwargs)
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

# ── Confirmation gate ─────────────────────────────────────────────────────────
#
# The assignment mandates a stateless microservice — all session state lives
# client-side.  So we can't store a "user confirmed" flag on the server.
#
# Instead the UI relays confirmation as a typed boolean on the payload
# (``ChatRequest.user_confirmed``).  Only an explicit, unambiguous user
# action (e.g. clicking "Confirm" on the dialog rendered after the LLM
# called ``request_user_confirmation``) sets that flag.  No free-text
# classification is performed — an LLM-fired ``submit_user_info`` tool call
# without the flag is rejected and the data is surfaced back to the UI as
# a pending confirmation, so the user can click Confirm instead.
#
# This keeps the gate deterministic and auditable: the submit step requires
# a user action, never a stochastic decision.  The check operates purely on
# the payload the client sends us, so statelessness is preserved.


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
    request_id: str,
) -> tuple[bool, str]:
    """Decide whether a submit_user_info tool call should be accepted.

    Returns (passed, reason).  ``reason`` is a short label used for logging
    so we can tell at-a-glance *why* the gate opened (or didn't).
    """
    if not req_user_confirmed:
        return False, "no_confirmation"

    # Cross-check ``confirmed_data`` (if provided) against the tool
    # arguments so the LLM can't quietly swap a field between the review
    # dialog and the final submit.
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


async def _handle_collection(
    client: AsyncAzureOpenAI,
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
      Gated by ``_confirmation_gate_passes``: requires the typed
      ``user_confirmed`` flag (and, if supplied, ``confirmed_data`` matching
      the tool arguments).
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
    response = await _call_llm(
        client,
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


async def _handle_qa(
    client: AsyncAzureOpenAI,
    retriever: Retriever,
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
    retrieval_query = _build_retrieval_query(messages)

    if retriever.is_ready():
        results = await retriever.search(retrieval_query, settings.retrieval_top_k)
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

    response = await _call_llm(client, openai_msgs, request_id)
    answer: str = response.choices[0].message.content or ""

    logger.info("[%s] Q&A answer generated (%d chars, mode=%s)", request_id, len(answer), mode)
    return ChatResponse(
        message=answer,
        phase="qa",
        request_id=request_id,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health_check(retriever: Retriever = Depends(get_retriever)):
    """Returns service status, knowledge-base readiness, and retriever status."""
    kb = get_knowledge_base()
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
async def chat(
    req: ChatRequest,
    client: AsyncAzureOpenAI = Depends(get_client),
    retriever: Retriever = Depends(get_retriever),
):
    start = time.perf_counter()
    logger.info(
        "POST /api/v1/chat | request_id=%s | phase=%s | msgs=%d",
        req.request_id,
        req.phase,
        len(req.messages),
    )

    if req.phase == "collection":
        result = await _handle_collection(
            client,
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
        result = await _handle_qa(
            client, retriever, req.messages, req.user_info, req.request_id
        )

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
