"""Extract Form 283 fields from OCR Markdown via one Azure OpenAI call."""

from __future__ import annotations

import json
import logging
import time

from openai import AzureOpenAI
from pydantic import ValidationError

from form_extraction.core.config import Settings, get_settings
from form_extraction.core.digits import override_fields, parse_date, parse_numeric
from form_extraction.core.prompts import build_messages
from form_extraction.core.schemas import DatePart, ExtractedForm, openai_json_schema

log = logging.getLogger("form_extraction.extractor")

_ADDRESS_FIELDS: frozenset[str] = frozenset(
    {"street", "houseNumber", "entrance", "apartment", "city", "postalCode", "poBox"}
)

_DATE_FIELDS: frozenset[str] = frozenset(
    {"dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic"}
)


def _apply_digit_corrections(form: ExtractedForm, markdown: str) -> ExtractedForm:
    """Replace LLM values with the anchor-label parser's reads where it succeeds."""
    top_updates: dict[str, object] = {}
    address_updates: dict[str, str] = {}

    for field in override_fields():
        if field in _DATE_FIELDS:
            parsed_date = parse_date(markdown, field)
            if parsed_date is None:
                continue
            dd, mm, yyyy = parsed_date
            current: DatePart = getattr(form, field)
            if current.day == dd and current.month == mm and current.year == yyyy:
                continue
            top_updates[field] = DatePart(day=dd, month=mm, year=yyyy)
            continue

        parsed = parse_numeric(markdown, field)
        if parsed is None:
            continue

        if field in _ADDRESS_FIELDS:
            if getattr(form.address, field) != parsed:
                address_updates[field] = parsed
        else:
            if getattr(form, field) != parsed:
                top_updates[field] = parsed

    if address_updates:
        top_updates["address"] = form.address.model_copy(update=address_updates)

    if not top_updates:
        return form
    log.info("extract.digit_fallback fields=%s", sorted(top_updates.keys()))
    return form.model_copy(update=top_updates)


def extract(markdown: str, settings: Settings | None = None) -> ExtractedForm:
    """Run one (or at most two) LLM calls and return an ExtractedForm."""
    s = settings or get_settings()
    if not s.azure_openai_endpoint or not s.azure_openai_key.get_secret_value():
        raise RuntimeError("Azure OpenAI is not configured.")

    schema = openai_json_schema()
    client = AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_key.get_secret_value(),
        api_version=s.azure_openai_api_version,
        timeout=s.request_timeout_s,
    )

    messages: list[dict[str, str]] = build_messages(markdown)

    log.info("extract.start md_chars=%d", len(markdown))
    t0 = time.perf_counter()
    payload = _call(client, s.azure_openai_deployment, messages, schema)

    try:
        form = ExtractedForm.model_validate(payload)
    except ValidationError as exc:
        log.warning("extract.retry errors=%d", exc.error_count())
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your previous JSON failed schema validation:\n{exc}\n"
                    "Return corrected JSON that matches the schema exactly. Output JSON only."
                ),
            }
        )
        payload = _call(client, s.azure_openai_deployment, messages, schema)
        form = ExtractedForm.model_validate(payload)

    form = _apply_digit_corrections(form, markdown)
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
