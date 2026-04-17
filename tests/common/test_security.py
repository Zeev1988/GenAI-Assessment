"""Tests for file-upload validation and PII redaction."""

from __future__ import annotations

import pytest
from common.errors import InputError
from common.security import mask_value, redact_payload, redact_text, validate_upload


def test_pdf_magic_is_accepted() -> None:
    assert validate_upload(b"%PDF-1.5 blabla", "x.pdf", 10 * 1024 * 1024) == "application/pdf"


def test_jpeg_magic_is_accepted() -> None:
    assert validate_upload(b"\xff\xd8\xff\xe0junk", "x.jpg", 10 * 1024 * 1024) == "image/jpeg"


def test_png_magic_is_accepted() -> None:
    assert validate_upload(b"\x89PNG\r\n\x1a\nthing", "x.png", 10 * 1024 * 1024) == "image/png"


def test_unknown_bytes_are_rejected() -> None:
    with pytest.raises(InputError):
        validate_upload(b"plain text", "notes.txt", 10 * 1024 * 1024)


def test_size_limit_enforced() -> None:
    with pytest.raises(InputError):
        validate_upload(b"%PDF-" + b"0" * (1024 * 1024), "x.pdf", 100)


def test_empty_upload_rejected() -> None:
    with pytest.raises(InputError):
        validate_upload(b"", "x.pdf", 100)


def test_redact_text_masks_ids_and_phones() -> None:
    text = "ID 123456789 phone 050-1234567"
    redacted = redact_text(text)
    assert "123456789" not in redacted
    assert "REDACTED_ID" in redacted
    assert "REDACTED_PHONE" in redacted


def test_redact_payload_masks_sensitive_keys() -> None:
    payload = {
        "idNumber": "123456789",
        "mobilePhone": "0501234567",
        "address": {"city": "תל אביב"},
    }
    masked = redact_payload(payload)
    assert masked["idNumber"].endswith("89")
    assert "*" in masked["idNumber"]
    assert masked["address"]["city"] == "תל אביב"


def test_mask_value_short_string() -> None:
    assert mask_value("12") == "**"
    assert mask_value("") == "**"
    assert mask_value("abcdef").endswith("ef")
