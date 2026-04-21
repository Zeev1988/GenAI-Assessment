"""Basic schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from form_extraction.core.schemas import ExtractedForm


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
