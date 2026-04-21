"""Azure Document Intelligence wrapper (prebuilt-layout, Markdown output)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential

from form_extraction.core.config import Settings, get_settings

log = logging.getLogger("form_extraction.ocr")


@dataclass
class OCRResult:
    markdown: str


def run_ocr(data: bytes, settings: Settings | None = None) -> OCRResult:
    s = settings or get_settings()
    if not s.azure_doc_intelligence_endpoint or not s.azure_doc_intelligence_key.get_secret_value():
        raise RuntimeError("Azure Document Intelligence is not configured.")

    log.info("ocr.start bytes=%d", len(data))
    t0 = time.perf_counter()

    client = DocumentIntelligenceClient(
        endpoint=s.azure_doc_intelligence_endpoint,
        credential=AzureKeyCredential(s.azure_doc_intelligence_key.get_secret_value()),
    )
    with client:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=data),
            locale="he",
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not result.pages:
        raise RuntimeError("OCR returned no pages; the document may be blank or unreadable.")

    markdown = (result.content or "").strip()
    log.info("ocr.done md_chars=%d elapsed_ms=%d", len(markdown), elapsed_ms)
    return OCRResult(markdown=markdown)
