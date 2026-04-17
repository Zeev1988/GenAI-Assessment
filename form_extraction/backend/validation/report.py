"""Build a validation report (completeness + issues) for an extracted form."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from form_extraction.backend.schemas import ExtractedForm
from form_extraction.backend.validation.rules import Issue, validate


@dataclass(slots=True)
class ValidationReport:
    completeness: float
    filled_fields: int
    total_fields: int
    issues: list[Issue] = field(default_factory=list)
    judge_score: int | None = None
    judge_comments: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": round(self.completeness, 4),
            "filled_fields": self.filled_fields,
            "total_fields": self.total_fields,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {"field": i.field, "severity": i.severity, "message": i.message}
                for i in self.issues
            ],
            "judge_score": self.judge_score,
            "judge_comments": list(self.judge_comments),
        }


def _count_fields(obj: Any) -> tuple[int, int]:
    """Return ``(filled, total)`` leaf counts."""
    if isinstance(obj, dict):
        filled = 0
        total = 0
        for value in obj.values():
            f, t = _count_fields(value)
            filled += f
            total += t
        return filled, total
    if isinstance(obj, list):
        filled = 0
        total = 0
        for item in obj:
            f, t = _count_fields(item)
            filled += f
            total += t
        return filled, total
    if isinstance(obj, str):
        return (1 if obj.strip() else 0), 1
    return 0, 0


def build_report(form: ExtractedForm) -> ValidationReport:
    data = form.model_dump()
    filled, total = _count_fields(data)
    completeness = (filled / total) if total else 0.0
    issues = validate(form)
    return ValidationReport(
        completeness=completeness,
        filled_fields=filled,
        total_fields=total,
        issues=issues,
    )
