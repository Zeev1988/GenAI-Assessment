"""Async Azure Document Intelligence client wrapper.

Uses the ``prebuilt-layout`` model which returns pages, tables, selection
marks, and a Markdown-style content string that preserves reading order for
both RTL (Hebrew) and LTR (English) text.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from common.clients.retry import run_with_retry
from common.config import Settings, get_settings
from common.errors import AzureAuthError, OCRError
from common.logging_config import get_logger

_log = get_logger(__name__)


@asynccontextmanager
async def _make_client(settings: Settings) -> AsyncIterator[DocumentIntelligenceClient]:
    settings.require_azure_di()
    client = DocumentIntelligenceClient(
        endpoint=settings.azure_doc_intelligence_endpoint,
        credential=AzureKeyCredential(settings.azure_doc_intelligence_key.get_secret_value()),
        api_version=settings.azure_doc_intelligence_api_version,
    )
    try:
        yield client
    finally:
        await client.close()


async def analyze_layout(
    data: bytes,
    *,
    settings: Settings | None = None,
    timeout_s: float | None = None,
) -> AnalyzeResult:
    """Run the ``prebuilt-layout`` model on ``data`` and return the full result.

    The call is retried on transient failures and surfaces typed errors on
    auth or service failures.
    """
    settings = settings or get_settings()
    timeout_s = timeout_s or settings.app_request_timeout_s

    started = time.perf_counter()
    try:
        async with _make_client(settings) as client:

            async def _invoke() -> AnalyzeResult:
                poller = await client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=AnalyzeDocumentRequest(bytes_source=data),
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
                return await poller.result()

            result: AnalyzeResult = await run_with_retry(_invoke)
    except ClientAuthenticationError as exc:
        raise AzureAuthError(
            "Azure Document Intelligence rejected the provided credentials.",
            details={"message": str(exc)},
        ) from exc
    except HttpResponseError as exc:
        raise OCRError(
            "Azure Document Intelligence request failed.",
            details={"status": exc.status_code, "message": exc.message},
        ) from exc

    duration_ms = (time.perf_counter() - started) * 1000.0
    pages = len(result.pages or [])
    _log.info(
        "ocr.complete",
        duration_ms=round(duration_ms, 2),
        pages=pages,
        tables=len(result.tables or []),
        content_chars=len(result.content or ""),
    )
    return result
