"""Schema round-trip + OpenAI json_schema compatibility."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from form_extraction.core.schemas import (
    HEBREW_KEY_MAP,
    ExtractedForm,
    openai_json_schema,
    to_hebrew_keys,
)


def test_default_form_is_all_empty_strings() -> None:
    data = ExtractedForm().model_dump()
    assert data["lastName"] == ""
    assert data["dateOfBirth"] == {"day": "", "month": "", "year": ""}
    assert data["address"]["city"] == ""


def test_round_trip_preserves_values() -> None:
    payload = {
        "lastName": "Cohen",
        "firstName": "Dana",
        "idNumber": "123456789",
        "gender": "נקבה",
        "dateOfBirth": {"day": "01", "month": "02", "year": "1990"},
        "address": {
            "street": "Herzl",
            "houseNumber": "10",
            "entrance": "",
            "apartment": "",
            "city": "Tel Aviv",
            "postalCode": "6100000",
            "poBox": "",
        },
        "landlinePhone": "",
        "mobilePhone": "0501234567",
        "jobType": "Engineer",
        "dateOfInjury": {"day": "03", "month": "04", "year": "2024"},
        "timeOfInjury": "09:30",
        "accidentLocation": "במפעל",
        "accidentAddress": "",
        "accidentDescription": "Slipped",
        "injuredBodyPart": "back",
        "signature": "Dana Cohen",
        "formFillingDate": {"day": "04", "month": "04", "year": "2024"},
        "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
        "medicalInstitutionFields": {
            "healthFundMember": "מכבי",
            "natureOfAccident": "",
            "medicalDiagnoses": "",
        },
    }
    assert ExtractedForm.model_validate(payload).model_dump() == payload


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedForm.model_validate({"unknown_key": "x"})


def test_openai_json_schema_is_strict() -> None:
    schema = openai_json_schema()

    def walk(node: dict) -> None:
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False
            props = set((node.get("properties") or {}).keys())
            required = set(node.get("required") or [])
            assert props == required
        for v in node.values():
            if isinstance(v, dict):
                walk(v)

    walk(schema)


def test_hebrew_key_conversion_is_deep_and_covers_top_level() -> None:
    hebrew = to_hebrew_keys(ExtractedForm().model_dump())
    assert "שם משפחה" in hebrew
    assert "יום" in hebrew["תאריך לידה"]
    assert set(ExtractedForm.model_fields.keys()) <= set(HEBREW_KEY_MAP.keys())
