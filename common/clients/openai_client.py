"""Async Azure OpenAI client wrapper with retries and token usage logging."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from openai import (
    AsyncAzureOpenAI,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
)

from common.clients.retry import run_with_retry
from common.config import Settings, get_settings
from common.errors import AzureAuthError, ExtractionError
from common.logging_config import get_logger

_log = get_logger(__name__)


@asynccontextmanager
async def _make_client(settings: Settings) -> AsyncIterator[AsyncAzureOpenAI]:
    settings.require_azure_openai()
    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_key.get_secret_value(),
        api_version=settings.azure_openai_api_version,
        timeout=settings.app_request_timeout_s,
    )
    try:
        yield client
    finally:
        await client.close()


async def chat_json(
    *,
    deployment: str,
    messages: list[dict[str, Any]],
    json_schema: dict[str, Any] | None = None,
    schema_name: str = "response",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    settings: Settings | None = None,
    stage: str = "chat",
) -> dict[str, Any]:
    """Call a chat completion that returns a JSON object.

    If ``json_schema`` is provided, the call uses Azure OpenAI's strict
    ``json_schema`` response format; otherwise it falls back to ``json_object``.

    Returns the parsed JSON. Retries are applied for transient failures.
    Token usage and latency are logged.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    if json_schema is not None:
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            },
        }
    else:
        response_format = {"type": "json_object"}

    try:
        async with _make_client(settings) as client:

            async def _invoke() -> Any:
                kwargs: dict[str, Any] = {
                    "model": deployment,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": response_format,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                return await client.chat.completions.create(**kwargs)

            response: Any = await run_with_retry(_invoke)
    except AuthenticationError as exc:
        raise AzureAuthError(
            "Azure OpenAI rejected the provided credentials.",
            details={"message": str(exc)},
        ) from exc
    except BadRequestError as exc:
        raise ExtractionError(
            "Azure OpenAI rejected the request.",
            details={"message": str(exc)},
        ) from exc
    except OpenAIError as exc:
        raise ExtractionError(
            "Azure OpenAI call failed.",
            details={"message": str(exc)},
        ) from exc

    duration_ms = (time.perf_counter() - started) * 1000.0
    usage = getattr(response, "usage", None)

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise ExtractionError("Azure OpenAI returned an empty response.")

    try:
        parsed: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        _log.error("openai.invalid_json", stage=stage, preview=content[:200])
        raise ExtractionError(
            "Azure OpenAI returned content that is not valid JSON.",
            details={"error": str(exc)},
        ) from exc

    _log.info(
        "openai.complete",
        stage=stage,
        deployment=deployment,
        duration_ms=round(duration_ms, 2),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        finish_reason=response.choices[0].finish_reason,
    )
    return parsed
