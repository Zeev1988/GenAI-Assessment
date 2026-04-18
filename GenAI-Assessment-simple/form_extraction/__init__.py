"""Form 283 field extraction library.

The `core` subpackage holds the framework-agnostic pipeline (OCR, LLM
extraction, Pydantic schema, format validation). The `ui` subpackage holds
the Streamlit entrypoint. Only `ui` may import from `core`; never the
other way around.
"""

from form_extraction.core.pipeline import PipelineResult, run
from form_extraction.core.schemas import ExtractedForm, to_hebrew_keys
from form_extraction.core.validate import Issue, ValidationReport

__all__ = [
    "ExtractedForm",
    "Issue",
    "PipelineResult",
    "ValidationReport",
    "run",
    "to_hebrew_keys",
]
