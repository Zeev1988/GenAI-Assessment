"""Schema and JSON-schema normalization tests."""

from __future__ import annotations

import pytest
from form_extraction.backend.schemas import (
    HEBREW_KEY_MAP,
    ExtractedForm,
    build_extraction_json_schema,
    to_hebrew_keys,
)
from pydantic import ValidationError


def _walk_required(node: dict) -> None:
    """Every object node must list every property in ``required``."""
    if node.get("type") == "object" or "properties" in node:
        props = set((node.get("properties") or {}).keys())
        required = set(node.get("required") or [])
        assert props == required, f"required mismatch at {node.get('title', node)}"
        assert node.get("additionalProperties") is False

    for value in node.values():
        if isinstance(value, dict):
            _walk_required(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _walk_required(item)


def test_default_form_is_all_empty_strings() -> None:
    form = ExtractedForm()
    data = form.model_dump()
    assert data["lastName"] == ""
    assert data["dateOfBirth"] == {"day": "", "month": "", "year": ""}
    assert data["address"]["city"] == ""
    assert data["medicalInstitutionFields"]["healthFundMember"] == ""


def test_round_trip_preserves_values() -> None:
    payload = {
        "lastName": "כהן",
        "firstName": "דנה",
        "idNumber": "123456789",
        "gender": "נקבה",
        "dateOfBirth": {"day": "01", "month": "02", "year": "1990"},
        "address": {
            "street": "הרצל",
            "houseNumber": "10",
            "entrance": "",
            "apartment": "",
            "city": "תל אביב",
            "postalCode": "6100000",
            "poBox": "",
        },
        "landlinePhone": "",
        "mobilePhone": "0501234567",
        "jobType": "מתכנת",
        "dateOfInjury": {"day": "03", "month": "04", "year": "2024"},
        "timeOfInjury": "09:30",
        "accidentLocation": "משרד",
        "accidentAddress": "הרצל 10",
        "accidentDescription": "נפל מכיסא",
        "injuredBodyPart": "גב",
        "signature": "",
        "formFillingDate": {"day": "04", "month": "04", "year": "2024"},
        "formReceiptDateAtClinic": {"day": "05", "month": "04", "year": "2024"},
        "medicalInstitutionFields": {
            "healthFundMember": "מכבי",
            "natureOfAccident": "נפילה",
            "medicalDiagnoses": "חבלה בגב",
        },
    }
    form = ExtractedForm.model_validate(payload)
    assert form.model_dump() == payload


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedForm.model_validate({"unknown": "x"})


def test_json_schema_is_openai_compatible() -> None:
    schema = build_extraction_json_schema()
    _walk_required(schema)


def test_hebrew_key_conversion_is_deep() -> None:
    data = ExtractedForm().model_dump()
    hebrew = to_hebrew_keys(data)
    assert "שם משפחה" in hebrew
    assert "יום" in hebrew["תאריך לידה"]
    assert "מיקוד" in hebrew["כתובת"]
    assert "חבר בקופת חולים" in hebrew['למילוי ע"י המוסד הרפואי']


def test_every_top_level_field_has_hebrew_label() -> None:
    form_fields = set(ExtractedForm.model_fields.keys())
    missing = form_fields - set(HEBREW_KEY_MAP.keys())
    assert not missing, f"missing Hebrew mapping for: {missing}"
