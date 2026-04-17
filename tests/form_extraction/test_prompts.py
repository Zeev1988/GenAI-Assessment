"""Prompt rendering tests."""

from __future__ import annotations

from form_extraction.backend.extraction.prompts import (
    SYSTEM_PROMPT,
    render_reask_prompt,
    render_user_prompt,
)


def test_system_prompt_enforces_empty_string_rule() -> None:
    assert "empty string" in SYSTEM_PROMPT.lower()
    assert "Output JSON only" in SYSTEM_PROMPT


def test_user_prompt_embeds_language_hint_and_content() -> None:
    prompt = render_user_prompt("שלום world 123", "he")
    assert "he" in prompt
    assert "BEGIN OCR" in prompt
    assert "END OCR" in prompt
    assert "שלום world 123" in prompt


def test_reask_prompt_includes_validation_error() -> None:
    text = render_reask_prompt("idNumber: value is required")
    assert "idNumber" in text
    assert "JSON" in text
