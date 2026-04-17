"""Typed exception hierarchy shared across features."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all application-level errors."""

    code: str = "app_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class AzureAuthError(AppError):
    """Raised when Azure credentials are missing or rejected."""

    code = "azure_auth_error"


class OCRError(AppError):
    """Raised when OCR fails or returns unusable content."""

    code = "ocr_error"


class ExtractionError(AppError):
    """Raised when the LLM fails to return a valid extraction payload."""

    code = "extraction_error"


class ValidationError(AppError):
    """Raised when validation rules cannot be applied (not a per-field issue)."""

    code = "validation_error"


class InputError(AppError):
    """Raised when the uploaded file is rejected (size, mime, empty)."""

    code = "input_error"
