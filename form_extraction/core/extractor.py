"""Extract Form 283 fields from OCR text using Azure OpenAI structured outputs.

The OCR stage (:mod:`form_extraction.core.ocr`) produces a single
``=== FORM 283 SPATIAL EXTRACTION ===`` document containing every field
value pre-computed from Azure Document Intelligence polygon coordinates
(word bounding boxes for text fields, selection-mark polygons for checkboxes).

The LLM's job is minimal: copy every value in the spatial header verbatim
into the JSON schema, split DDMMYYYY dates into day/month/year parts, and
map checkbox labels to their enum values.
"""

from __future__ import annotations

import json
import logging
import time

from openai import AzureOpenAI
from pydantic import ValidationError

from form_extraction.core.config import Settings, get_settings
from form_extraction.core.prompts import build_messages
from form_extraction.core.schemas import ExtractedForm, openai_json_schema

log = logging.getLogger("form_extraction.extractor")


def extract(ocr_text: str, settings: Settings | None = None) -> ExtractedForm:
    """Return an ExtractedForm built from one (or at most two) LLM calls."""
    s = settings or get_settings()
    if not s.azure_openai_endpoint or not s.azure_openai_key.get_secret_value():
        raise RuntimeError(
            "Azure OpenAI is not configured. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY."
        )

    schema = openai_json_schema()
    client = AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_key.get_secret_value(),
        api_version=s.azure_openai_api_version,
        timeout=s.request_timeout_s,
    )

    messages: list[dict[str, str]] = build_messages(ocr_text)

    log.info("extract.start ocr_chars=%d", len(ocr_text))
    t0 = time.perf_counter()
    payload = _call(client, s.azure_openai_deployment, messages, schema)

    try:
        form = ExtractedForm.model_validate(payload)
    except ValidationError as exc:
        # One corrective re-ask on the rare case strict mode still produces
        # a payload Pydantic rejects (e.g., an enum value that slipped through).
        log.warning("extract.retry errors=%d", exc.error_count())
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous JSON failed schema validation:\n"
                    f"{exc}\n"
                    "Return a corrected JSON that matches the schema exactly. "
                    "Keep every field you had right; only fix the validation errors. "
                    "Output JSON only."
                ),
            }
        )
        payload = _call(client, s.azure_openai_deployment, messages, schema)
        form = ExtractedForm.model_validate(payload)
        log.info("extract.done retried=1 elapsed_ms=%d", int((time.perf_counter() - t0) * 1000))
        return form

    log.info("extract.done elapsed_ms=%d", int((time.perf_counter() - t0) * 1000))
    return form


def _call(
    client: AzureOpenAI,
    deployment: str,
    messages: list[dict[str, str]],
    schema: dict,
) -> dict:
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ExtractedForm", "schema": schema, "strict": True},
        },
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Azure OpenAI returned an empty response.")
    return json.loads(content)
