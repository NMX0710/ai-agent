import sys
from pathlib import Path
from uuid import uuid4

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.nutrition.apple_health_mapping import (
    meal_log_to_apple_health_payload,
    nutrition_totals_to_apple_health_samples,
)
from app.nutrition.meallog import CanonicalMealLog


def _meal_log_with_nutrients() -> CanonicalMealLog:
    return CanonicalMealLog.model_validate(
        {
            "schema_version": "meal_log.v1",
            "meal_id": str(uuid4()),
            "user_id": "tg:10001",
            "consumed_at": "2026-03-10T20:00:00-04:00",
            "timezone": "America/New_York",
            "meal_type": "dinner",
            "input_source": "text",
            "nutrition_totals": {
                "energy_kcal": 500,
                "protein_g": 35,
                "carbs_g": 50,
                "fat_g": 18,
                "sodium_mg": 1200,
                "cholesterol_mg": 300,
                "micronutrients": {
                    "vitamin_c_mg": {"value": 90, "unit": "mg"},
                    "HKQuantityTypeIdentifierDietaryVitaminE": {"value": 12, "unit": "mg"},
                },
            },
            "confidence": {"overall": 0.9},
            "raw_inputs": {"text": "chicken, rice and vegetables"},
        }
    )


def test_nutrition_totals_to_apple_health_samples_core_and_conversion():
    meal_log = _meal_log_with_nutrients()
    samples = nutrition_totals_to_apple_health_samples(meal_log.nutrition_totals)
    by_id = {s.identifier: s for s in samples}

    assert by_id["HKQuantityTypeIdentifierDietaryEnergyConsumed"].value == 500
    assert by_id["HKQuantityTypeIdentifierDietaryProtein"].value == 35
    assert by_id["HKQuantityTypeIdentifierDietarySodium"].value == 1.2
    assert by_id["HKQuantityTypeIdentifierDietaryCholesterol"].value == 0.3
    assert round(by_id["HKQuantityTypeIdentifierDietaryVitaminC"].value, 6) == 0.09
    assert round(by_id["HKQuantityTypeIdentifierDietaryVitaminE"].value, 6) == 0.012


def test_meal_log_to_apple_health_payload_shape():
    meal_log = _meal_log_with_nutrients()
    payload = meal_log_to_apple_health_payload(meal_log)

    assert payload["meal_type"] == "dinner"
    assert payload["timezone"] == "America/New_York"
    assert payload["food_type"] == "chicken, rice and vegetables"
    assert payload["sync_identifier"].startswith("meal:")
    assert payload["sync_version"] == 1
    assert len(payload["samples"]) >= 4
