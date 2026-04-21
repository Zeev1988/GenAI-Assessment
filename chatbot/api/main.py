"""Stateless FastAPI chatbot API.

The server holds no session state — every request carries the full
conversation history. Two phases:
  - collection: LLM gathers member details; `submit_user_info` tool call
    finalises and transitions to Q&A.
  - qa: LLM answers health-fund questions using retrieved chunks.
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
from pydantic import BaseModel, Field

from common import get_logger

from ..core.config import get_settings
from ..core.knowledge import get_knowledge_base
from ..core.prompts import (
    COLLECTION_SYSTEM_PROMPT,
    SUBMIT_USER_INFO_TOOL,
    build_qa_system_prompt,
)
from ..core.retrieval import Retriever

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level, log_file=settings.log_file)


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    phase: str = Field(..., pattern="^(collection|qa)$")
    messages: list[Message] = Field(default_factory=list)
    user_info: Optional[dict[str, Any]] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ChatResponse(BaseModel):
    message: str
    phase: str
    transition: bool = False
    extracted_user_info: Optional[dict[str, Any]] = None
    request_id: str
    processing_time_ms: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HMO Chatbot API starting up")
    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
        timeout=settings.request_timeout_s,
        max_retries=3,
    )
    retriever = Retriever(client, embedding_deployment=settings.azure_openai_embedding_deployment)
    app.state.openai_client = client
    app.state.retriever = retriever

    kb = get_knowledge_base()
    if kb.is_loaded():
        await retriever.index(kb.chunks())

    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="HMO Chatbot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client(request: Request) -> AsyncAzureOpenAI:
    return request.app.state.openai_client


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def _openai_messages(system_prompt: str, history: list[Message]) -> list[dict]:
    return [{"role": "system", "content": system_prompt}] + [
        {"role": m.role, "content": m.content} for m in history
    ]


async def _call_llm(
    client: AsyncAzureOpenAI,
    messages: list[dict],
    request_id: str,
    tools: Optional[list[dict]] = None,
) -> Any:
    kwargs: dict[str, Any] = {"model": settings.azure_openai_deployment, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        return await client.chat.completions.create(**kwargs)
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="The language model timed out.")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Cannot reach the language model.")
    except APIError as exc:
        logger.error("[%s] LLM API error: %s", request_id, exc)
        raise HTTPException(status_code=502, detail=f"Language model error: {exc}")


async def _handle_collection(
    client: AsyncAzureOpenAI,
    messages: list[Message],
    request_id: str,
) -> ChatResponse:
    # On first load (empty history) inject a hidden start signal so the LLM greets first.
    effective = messages or [Message(role="user", content="[SESSION_START]")]

    openai_msgs = _openai_messages(COLLECTION_SYSTEM_PROMPT, effective)
    response = await _call_llm(client, openai_msgs, request_id, tools=[SUBMIT_USER_INFO_TOOL])

    choice = response.choices[0]
    content = choice.message.content or ""
    tool_calls = choice.message.tool_calls

    if tool_calls and tool_calls[0].function.name == "submit_user_info":
        try:
            user_info = json.loads(tool_calls[0].function.arguments)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse tool arguments.")

        logger.info(
            "[%s] Collection complete — %s %s | %s | %s",
            request_id,
            user_info.get("first_name"),
            user_info.get("last_name"),
            user_info.get("hmo_name"),
            user_info.get("insurance_tier"),
        )

        if not content:
            fn = user_info.get("first_name", "")
            hmo = user_info.get("hmo_name", "")
            content = f"תודה {fn}! הפרטים נשמרו. כמבוטח/ת ב{hmo}, איך אוכל לעזור?"

        return ChatResponse(
            message=content,
            phase="qa",
            transition=True,
            extracted_user_info=user_info,
            request_id=request_id,
        )

    return ChatResponse(message=content, phase="collection", request_id=request_id)


async def _handle_qa(
    client: AsyncAzureOpenAI,
    retriever: Retriever,
    messages: list[Message],
    user_info: dict[str, Any],
    request_id: str,
) -> ChatResponse:
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    results = await retriever.search(last_user, settings.retrieval_top_k)

    if not results:
        raise HTTPException(status_code=503, detail="Knowledge base is not ready.")

    separator = "\n" + "=" * 80 + "\n"
    knowledge = separator.join(chunk.prompt_block for chunk, _ in results)

    system_prompt = build_qa_system_prompt(user_info, knowledge)
    openai_msgs = _openai_messages(system_prompt, messages)

    logger.info(
        "[%s] Q&A | HMO=%s tier=%s | %d chunks",
        request_id,
        user_info.get("hmo_name"),
        user_info.get("insurance_tier"),
        len(results),
    )

    response = await _call_llm(client, openai_msgs, request_id)
    return ChatResponse(
        message=response.choices[0].message.content or "",
        phase="qa",
        request_id=request_id,
    )


@app.get("/health")
async def health_check(retriever: Retriever = Depends(get_retriever)):
    kb = get_knowledge_base()
    return {
        "status": "ok",
        "knowledge_base_loaded": kb.is_loaded(),
        "topic_count": kb.topic_count(),
        "retrieval_ready": retriever.is_ready(),
        "indexed_chunks": retriever.chunk_count(),
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client: AsyncAzureOpenAI = Depends(get_client),
    retriever: Retriever = Depends(get_retriever),
):
    start = time.perf_counter()
    logger.info("POST /chat | %s | phase=%s | msgs=%d", req.request_id, req.phase, len(req.messages))

    if req.phase == "collection":
        result = await _handle_collection(client, req.messages, req.request_id)
    else:
        if not req.user_info:
            raise HTTPException(status_code=422, detail="user_info is required for Q&A.")
        result = await _handle_qa(client, retriever, req.messages, req.user_info, req.request_id)

    result.processing_time_ms = int((time.perf_counter() - start) * 1000)
    return result


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})


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
