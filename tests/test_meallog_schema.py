import sys
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.nutrition.meallog import CanonicalMealLog, MealLogStatus


def _valid_payload() -> dict:
    return {
        "schema_version": "meal_log.v1",
        "meal_id": str(uuid4()),
        "user_id": "tg:999000",
        "consumed_at": "2026-03-10T19:30:00-04:00",
        "timezone": "America/New_York",
        "meal_type": "dinner",
        "input_source": "mixed",
        "status": "draft",
        "record_version": 1,
        "nutrition_totals": {
            "energy_kcal": 620,
            "protein_g": 42,
            "carbs_g": 58,
            "fat_g": 22,
            "sodium_mg": 980,
        },
        "confidence": {"overall": 0.82},
    }


def test_meallog_v1_minimal_valid():
    obj = CanonicalMealLog.model_validate(_valid_payload())
    assert obj.schema_version == "meal_log.v1"
    assert obj.status == MealLogStatus.draft
    assert obj.nutrition_totals.energy_kcal == 620


def test_meallog_requires_timezone_aware_consumed_at():
    payload = _valid_payload()
    payload["consumed_at"] = "2026-03-10T19:30:00"
    with pytest.raises(ValidationError):
        CanonicalMealLog.model_validate(payload)


def test_meallog_rejects_invalid_schema_version():
    payload = _valid_payload()
    payload["schema_version"] = "meal_log.v2"
    with pytest.raises(ValidationError):
        CanonicalMealLog.model_validate(payload)


def test_meallog_rejects_unknown_extra_fields():
    payload = _valid_payload()
    payload["unknown_field"] = "unexpected"
    with pytest.raises(ValidationError):
        CanonicalMealLog.model_validate(payload)
