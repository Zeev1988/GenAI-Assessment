"""Field extraction using Azure OpenAI JSON-schema structured outputs."""

from __future__ import annotations

import json
from typing import Any

from common.clients.openai_client import chat_json
from common.config import Settings, get_settings
from common.errors import ExtractionError
from common.logging_config import get_logger
from pydantic import ValidationError as PydanticValidationError

from form_extraction.backend.extraction.prompts import (
    SYSTEM_PROMPT,
    render_reask_prompt,
    render_user_prompt,
)
from form_extraction.backend.ocr.layout import OCRResult
from form_extraction.backend.schemas import ExtractedForm, build_extraction_json_schema

_log = get_logger(__name__)


async def extract_fields(
    ocr: OCRResult,
    *,
    settings: Settings | None = None,
    max_attempts: int = 2,
) -> ExtractedForm:
    """Run the LLM extraction and return a validated :class:`ExtractedForm`.

    On Pydantic validation failure the function issues a single corrective
    re-ask with the validation error attached.
    """
    settings = settings or get_settings()
    schema = build_extraction_json_schema()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_user_prompt(ocr.content, ocr.language_hint),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        payload = await chat_json(
            deployment=settings.azure_openai_deployment_extract,
            messages=messages,
            json_schema=schema,
            schema_name="ExtractedForm",
            settings=settings,
            stage=f"extract_attempt_{attempt}",
        )
        try:
            return ExtractedForm.model_validate(payload)
        except PydanticValidationError as exc:
            last_error = exc
            _log.warning(
                "extract.validation_failed",
                attempt=attempt,
                error_count=len(exc.errors()),
            )
            if attempt >= max_attempts:
                break
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": render_reask_prompt(str(exc)),
                    },
                ]
            )

    raise ExtractionError(
        "LLM output failed schema validation after retries.",
        details={"error": str(last_error) if last_error else "unknown"},
    )
