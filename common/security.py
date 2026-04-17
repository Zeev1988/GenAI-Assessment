"""Security helpers: input validation and PII redaction for logs."""

from __future__ import annotations

import re
from typing import Any

from common.errors import InputError

_ALLOWED_MIMES: frozenset[str] = frozenset({"application/pdf", "image/jpeg", "image/png"})

_PDF_MAGIC = b"%PDF-"
_JPEG_MAGIC = (b"\xff\xd8\xff",)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sniff_mime(data: bytes, filename: str | None = None) -> str:
    """Best-effort MIME sniffing using magic bytes + filename hint.

    Returns a concrete MIME string or raises :class:`InputError` if the content
    is not in the allow-list.
    """
    if not data:
        raise InputError("Uploaded file is empty.")

    if data.startswith(_PDF_MAGIC):
        return "application/pdf"
    if any(data.startswith(m) for m in _JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"

    if filename:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return "application/pdf"
        if lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if lower.endswith(".png"):
            return "image/png"

    raise InputError(
        "Unsupported file type. Only PDF, JPEG, and PNG uploads are allowed.",
        details={"sniff": "unknown"},
    )


def validate_upload(data: bytes, filename: str | None, max_bytes: int) -> str:
    """Validate size and MIME of an upload; return the detected MIME type."""
    if len(data) > max_bytes:
        raise InputError(
            "File too large.",
            details={"size_bytes": len(data), "max_bytes": max_bytes},
        )
    mime = sniff_mime(data, filename)
    if mime not in _ALLOWED_MIMES:
        raise InputError("Unsupported MIME type.", details={"mime": mime})
    return mime


# --- PII redaction ---------------------------------------------------------

_ID_RE = re.compile(r"\b\d{9}\b")
_PHONE_RE = re.compile(r"\b0\d{1,2}[-\s]?\d{7}\b")


def redact_text(text: str) -> str:
    """Mask 9-digit IDs and Israeli phone numbers in free-form text."""
    if not text:
        return text
    text = _ID_RE.sub("[REDACTED_ID]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "idnumber",
        "id_number",
        "idNumber",
        "landlinePhone",
        "mobilePhone",
        "landline_phone",
        "mobile_phone",
        "signature",
    }
)


def redact_payload(obj: Any) -> Any:
    """Return a deep copy of ``obj`` with sensitive field values masked.

    The original object is not modified. Useful before logging extracted JSON.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in _SENSITIVE_KEYS and isinstance(value, str) and value:
                out[key] = mask_value(value)
            else:
                out[key] = redact_payload(value)
        return out
    if isinstance(obj, list):
        return [redact_payload(item) for item in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def mask_value(value: str) -> str:
    """Mask a string, preserving only the last two characters."""
    if len(value) <= 2:
        return "**"
    return "*" * (len(value) - 2) + value[-2:]
