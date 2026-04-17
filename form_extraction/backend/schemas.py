"""Pydantic v2 schemas matching the assignment's exact JSON shape.

The schema is the single source of truth: the same model drives
(a) the LLM's JSON-schema structured output, (b) parsing/validation,
(c) UI rendering, and (d) regression (golden) tests.

All fields default to empty strings so that "missing" data is represented as
`""` per the assignment, without forcing the LLM to emit nulls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """Base model with strict, deterministic behavior."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Nested date / address / medical blocks
# ---------------------------------------------------------------------------


class DatePart(_StrictModel):
    """A split date with string fields (as per the required output)."""

    day: str = Field(default="")
    month: str = Field(default="")
    year: str = Field(default="")


class Address(_StrictModel):
    street: str = Field(default="")
    houseNumber: str = Field(default="")
    entrance: str = Field(default="")
    apartment: str = Field(default="")
    city: str = Field(default="")
    postalCode: str = Field(default="")
    poBox: str = Field(default="")


class MedicalInstitutionFields(_StrictModel):
    healthFundMember: str = Field(default="")
    natureOfAccident: str = Field(default="")
    medicalDiagnoses: str = Field(default="")


# ---------------------------------------------------------------------------
# Top-level extracted form
# ---------------------------------------------------------------------------


class ExtractedForm(_StrictModel):
    """Top-level JSON shape for the National Insurance Institute Form 283."""

    lastName: str = Field(default="")
    firstName: str = Field(default="")
    idNumber: str = Field(default="")
    gender: str = Field(default="")
    dateOfBirth: DatePart = Field(default_factory=DatePart)
    address: Address = Field(default_factory=Address)
    landlinePhone: str = Field(default="")
    mobilePhone: str = Field(default="")
    jobType: str = Field(default="")
    dateOfInjury: DatePart = Field(default_factory=DatePart)
    timeOfInjury: str = Field(default="")
    accidentLocation: str = Field(default="")
    accidentAddress: str = Field(default="")
    accidentDescription: str = Field(default="")
    injuredBodyPart: str = Field(default="")
    signature: str = Field(default="")
    formFillingDate: DatePart = Field(default_factory=DatePart)
    formReceiptDateAtClinic: DatePart = Field(default_factory=DatePart)
    medicalInstitutionFields: MedicalInstitutionFields = Field(
        default_factory=MedicalInstitutionFields
    )


# ---------------------------------------------------------------------------
# Hebrew-keyed variant for the optional Hebrew output mode.
# ---------------------------------------------------------------------------

#: Mapping from the canonical English keys (used internally) to the Hebrew
#: labels from the assignment. We transform both ways instead of defining a
#: parallel Pydantic model to keep a single source of truth.
HEBREW_KEY_MAP: dict[str, str] = {
    "lastName": "שם משפחה",
    "firstName": "שם פרטי",
    "idNumber": "מספר זהות",
    "gender": "מין",
    "dateOfBirth": "תאריך לידה",
    "address": "כתובת",
    "landlinePhone": "טלפון קווי",
    "mobilePhone": "טלפון נייד",
    "jobType": "סוג העבודה",
    "dateOfInjury": "תאריך הפגיעה",
    "timeOfInjury": "שעת הפגיעה",
    "accidentLocation": "מקום התאונה",
    "accidentAddress": "כתובת מקום התאונה",
    "accidentDescription": "תיאור התאונה",
    "injuredBodyPart": "האיבר שנפגע",
    "signature": "חתימה",
    "formFillingDate": "תאריך מילוי הטופס",
    "formReceiptDateAtClinic": "תאריך קבלת הטופס בקופה",
    "medicalInstitutionFields": 'למילוי ע"י המוסד הרפואי',
    "street": "רחוב",
    "houseNumber": "מספר בית",
    "entrance": "כניסה",
    "apartment": "דירה",
    "city": "ישוב",
    "postalCode": "מיקוד",
    "poBox": "תא דואר",
    "day": "יום",
    "month": "חודש",
    "year": "שנה",
    "healthFundMember": "חבר בקופת חולים",
    "natureOfAccident": "מהות התאונה",
    "medicalDiagnoses": "אבחנות רפואיות",
}


def to_hebrew_keys(obj: Any) -> Any:
    """Deep-convert a mapping with English keys to Hebrew-labeled keys."""
    if isinstance(obj, dict):
        return {HEBREW_KEY_MAP.get(k, k): to_hebrew_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_hebrew_keys(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# JSON schema used for Azure OpenAI structured outputs.
# ---------------------------------------------------------------------------


def build_extraction_json_schema() -> dict[str, Any]:
    """Return a JSON schema suitable for Azure OpenAI ``response_format``.

    Azure OpenAI ``json_schema`` mode imposes additional constraints compared
    to Pydantic's default schema:

    * every object must declare ``additionalProperties: false``;
    * every property must appear in ``required``;
    * ``default`` and a few other draft-specific keywords are not allowed.

    We start from Pydantic's schema and normalize it.
    """
    schema = ExtractedForm.model_json_schema()
    _normalize_for_openai(schema)
    for definition in schema.get("$defs", {}).values():
        _normalize_for_openai(definition)
    return schema


_STRIP_KEYWORDS: frozenset[str] = frozenset({"default", "title", "examples", "description"})


def _normalize_for_openai(node: Any) -> None:
    """Recursively adapt a JSON schema node to Azure OpenAI's strict mode."""
    if isinstance(node, dict):
        for key in list(node.keys()):
            if key in _STRIP_KEYWORDS:
                node.pop(key, None)

        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())

        for value in node.values():
            _normalize_for_openai(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_for_openai(item)
