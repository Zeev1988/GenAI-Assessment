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
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AzureOpenAI, APIError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field, field_validator

from common import get_logger

from ..core.config import get_settings
from ..core.knowledge import get_knowledge_base
from ..core.prompts import (
    COLLECTION_SYSTEM_PROMPT,
    SUBMIT_USER_INFO_TOOL,
    build_qa_system_prompt,
)

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
    request_id: str
    processing_time_ms: int = Field(default=0)


# ── Application lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the knowledge base before accepting requests."""
    logger.info("=== HMO Chatbot API starting up ===")
    kb = get_knowledge_base()
    if kb.is_loaded():
        logger.info(
            "Knowledge base ready: %d topics loaded", kb.topic_count()
        )
    else:
        logger.warning(
            "Knowledge base NOT loaded — Q&A answers will be unavailable."
        )
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

def _handle_collection(
    messages: list[Message],
    request_id: str,
) -> ChatResponse:
    """
    Drive the information-collection phase.

    The LLM is given the ``submit_user_info`` tool.  When it calls that tool
    the collection phase is complete and we transition to Q&A.
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
            "[%s] Collection phase — %d messages in history",
            request_id,
            len(messages),
        )

    openai_msgs = _openai_messages(COLLECTION_SYSTEM_PROMPT, effective_messages)
    response = _call_llm(openai_msgs, request_id, tools=[SUBMIT_USER_INFO_TOOL])

    choice = response.choices[0]
    content: str = choice.message.content or ""
    tool_calls = choice.message.tool_calls

    # ── Tool call → collection complete ───────────────────────────────────────
    if tool_calls:
        tc = tool_calls[0]
        if tc.function.name == "submit_user_info":
            try:
                user_info: dict = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                logger.error(
                    "[%s] Cannot parse submit_user_info arguments: %s", request_id, exc
                )
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse collected user information.",
                )

            logger.info(
                "[%s] Collection complete — %s %s | HMO: %s | tier: %s",
                request_id,
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

    # ── Normal dialogue turn ───────────────────────────────────────────────────
    logger.info("[%s] Collection dialogue — reply: %d chars", request_id, len(content))
    return ChatResponse(
        message=content,
        phase="collection",
        request_id=request_id,
    )


def _handle_qa(
    messages: list[Message],
    user_info: dict[str, Any],
    request_id: str,
) -> ChatResponse:
    """Answer health-fund questions using the member's profile + knowledge base."""
    kb = get_knowledge_base()
    knowledge_content = kb.all_content()

    if not knowledge_content:
        logger.warning("[%s] Knowledge base is empty — answers will be limited", request_id)

    system_prompt = build_qa_system_prompt(user_info, knowledge_content)
    openai_msgs = _openai_messages(system_prompt, messages)

    logger.info(
        "[%s] Q&A phase — %s | HMO: %s | tier: %s | %d messages",
        request_id,
        user_info.get("first_name"),
        user_info.get("hmo_name"),
        user_info.get("insurance_tier"),
        len(messages),
    )

    response = _call_llm(openai_msgs, request_id)
    answer: str = response.choices[0].message.content or ""

    logger.info("[%s] Q&A answer generated (%d chars)", request_id, len(answer))
    return ChatResponse(
        message=answer,
        phase="qa",
        request_id=request_id,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health_check():
    """Returns service status and knowledge-base readiness."""
    kb = get_knowledge_base()
    return {
        "status": "ok",
        "knowledge_base_loaded": kb.is_loaded(),
        "topic_count": kb.topic_count(),
        "topics": kb.topic_titles(),
        "deployment": settings.azure_openai_deployment,
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
        result = _handle_collection(req.messages, req.request_id)
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
