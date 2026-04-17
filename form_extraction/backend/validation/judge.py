"""Optional LLM-as-judge validator using GPT-4o-mini."""

from __future__ import annotations

import json
from typing import Any

from common.clients.openai_client import chat_json
from common.config import Settings, get_settings
from common.logging_config import get_logger

from form_extraction.backend.schemas import ExtractedForm

_log = get_logger(__name__)


_JUDGE_SYSTEM = """\
You are an impartial evaluator. You compare OCR text with an extracted JSON \
payload and judge how faithfully the payload represents the OCR. Do not \
reward or punish for missing data that is genuinely absent from the OCR. \
Return JSON only.
"""


_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {
            "type": "integer",
            "description": "0 (unfaithful) .. 100 (perfectly faithful).",
        },
        "comments": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short, specific observations about mismatches.",
        },
    },
    "required": ["score", "comments"],
}


async def judge_extraction(
    ocr_content: str,
    form: ExtractedForm,
    *,
    settings: Settings | None = None,
) -> tuple[int, list[str]]:
    """Return ``(score, comments)`` from the LLM judge."""
    settings = settings or get_settings()

    user_payload = (
        "OCR:\n----- BEGIN -----\n"
        f"{ocr_content}\n"
        "----- END -----\n\n"
        "Extracted JSON:\n"
        f"{json.dumps(form.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        "Score faithfulness from 0 to 100 and list up to 5 concise comments."
    )
    try:
        result = await chat_json(
            deployment=settings.azure_openai_deployment_judge,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            json_schema=_JUDGE_SCHEMA,
            schema_name="JudgeResult",
            settings=settings,
            stage="judge",
        )
    except Exception as exc:
        _log.warning("judge.failed", error=str(exc))
        return 0, [f"judge unavailable: {exc}"]

    score_raw = result.get("score", 0)
    try:
        score = max(0, min(100, int(score_raw)))
    except (TypeError, ValueError):
        score = 0

    comments = [str(c) for c in result.get("comments", []) if c]
    return score, comments
