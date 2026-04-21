"""Pydantic models for Form 283. camelCase keys match the assignment spec.

`schemas.py` is the single source of truth for the three checkbox enums
(gender, health fund, accident location). The extractor prompt and tests
import the label tuples so allowed values cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GENDER_LABELS: tuple[str, ...] = ("זכר", "נקבה")
HEALTH_FUND_LABELS: tuple[str, ...] = ("כללית", "מכבי", "מאוחדת", "לאומית")
ACCIDENT_LOCATION_LABELS: tuple[str, ...] = (
    "במפעל",
    "ת. דרכים בעבודה",
    "ת. דרכים בדרך לעבודה/מהעבודה",
    "תאונה בדרך ללא רכב",
    "אחר",
)

# Literal unions include "" so "no checkbox selected" is valid.
GenderLabel = Literal["זכר", "נקבה", ""]
HealthFundLabel = Literal["כללית", "מכבי", "מאוחדת", "לאומית", ""]
AccidentLocationLabel = Literal[
    "במפעל",
    "ת. דרכים בעבודה",
    "ת. דרכים בדרך לעבודה/מהעבודה",
    "תאונה בדרך ללא רכב",
    "אחר",
    "",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DatePart(_Model):
    day: str = Field(default="")
    month: str = Field(default="")
    year: str = Field(default="")


class Address(_Model):
    street: str = Field(default="")
    houseNumber: str = Field(default="")
    entrance: str = Field(default="")
    apartment: str = Field(default="")
    city: str = Field(default="")
    postalCode: str = Field(default="")
    poBox: str = Field(default="")


class MedicalInstitutionFields(_Model):
    healthFundMember: HealthFundLabel = Field(default="")
    natureOfAccident: str = Field(default="")
    medicalDiagnoses: str = Field(default="")


class ExtractedForm(_Model):
    lastName: str = Field(default="")
    firstName: str = Field(default="")
    idNumber: str = Field(default="")
    gender: GenderLabel = Field(default="")
    dateOfBirth: DatePart = Field(default_factory=DatePart)
    address: Address = Field(default_factory=Address)
    landlinePhone: str = Field(default="")
    mobilePhone: str = Field(default="")
    jobType: str = Field(default="")
    dateOfInjury: DatePart = Field(default_factory=DatePart)
    timeOfInjury: str = Field(default="")
    accidentLocation: AccidentLocationLabel = Field(default="")
    accidentAddress: str = Field(default="")
    accidentDescription: str = Field(default="")
    injuredBodyPart: str = Field(default="")
    signature: str = Field(default="")
    formFillingDate: DatePart = Field(default_factory=DatePart)
    formReceiptDateAtClinic: DatePart = Field(default_factory=DatePart)
    medicalInstitutionFields: MedicalInstitutionFields = Field(
        default_factory=MedicalInstitutionFields
    )


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
    """Deep-rename dict keys from English to Hebrew labels."""
    if isinstance(obj, dict):
        return {HEBREW_KEY_MAP.get(k, k): to_hebrew_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_hebrew_keys(v) for v in obj]
    return obj


def openai_json_schema() -> dict[str, Any]:
    """JSON schema for Azure OpenAI's `json_schema` response_format (strict mode)."""
    schema = ExtractedForm.model_json_schema()
    _normalize(schema)
    for defn in schema.get("$defs", {}).values():
        _normalize(defn)
    return schema


# Keys Azure strict mode does not want.
_STRIP = frozenset({"default", "title", "examples", "description"})


def _normalize(node: Any) -> None:
    if isinstance(node, dict):
        for k in list(node.keys()):
            if k in _STRIP:
                node.pop(k, None)
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            props = node.get("properties")
            if isinstance(props, dict):
                node["required"] = list(props.keys())
        for v in node.values():
            _normalize(v)
    elif isinstance(node, list):
        for item in node:
            _normalize(item)
