"""Framework-agnostic extraction pipeline.

Nothing in this package imports from `form_extraction.ui`. Keep it that way.
"""

from form_extraction.core.pipeline import PipelineResult, run
from form_extraction.core.schemas import (
    ACCIDENT_LOCATION_LABELS,
    GENDER_LABELS,
    HEALTH_FUND_LABELS,
    HEBREW_KEY_MAP,
    ExtractedForm,
    openai_json_schema,
    to_hebrew_keys,
)
from form_extraction.core.validate import Issue, ValidationReport, validate

__all__ = [
    "ACCIDENT_LOCATION_LABELS",
    "ExtractedForm",
    "GENDER_LABELS",
    "HEALTH_FUND_LABELS",
    "HEBREW_KEY_MAP",
    "Issue",
    "PipelineResult",
    "ValidationReport",
    "openai_json_schema",
    "run",
    "to_hebrew_keys",
    "validate",
]
