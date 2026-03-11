from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.nutrition.meallog import CanonicalMealLog, NutritionTotals


@dataclass(frozen=True)
class AppleHealthQuantitySample:
    identifier: str
    value: float
    unit: str


# Canonical field -> (HK quantity identifier, output unit)
_CANONICAL_TO_HK: Dict[str, Tuple[str, str]] = {
    "energy_kcal": ("HKQuantityTypeIdentifierDietaryEnergyConsumed", "kcal"),
    "protein_g": ("HKQuantityTypeIdentifierDietaryProtein", "g"),
    "carbs_g": ("HKQuantityTypeIdentifierDietaryCarbohydrates", "g"),
    "fat_g": ("HKQuantityTypeIdentifierDietaryFatTotal", "g"),
    "fiber_g": ("HKQuantityTypeIdentifierDietaryFiber", "g"),
    "sugar_g": ("HKQuantityTypeIdentifierDietarySugar", "g"),
    "sodium_mg": ("HKQuantityTypeIdentifierDietarySodium", "g"),
    "cholesterol_mg": ("HKQuantityTypeIdentifierDietaryCholesterol", "g"),
    "water_ml": ("HKQuantityTypeIdentifierDietaryWater", "mL"),
}


# Optional micronutrient aliases kept platform-neutral in canonical layer.
_MICRONUTRIENT_ALIAS_TO_HK: Dict[str, Tuple[str, str]] = {
    "calcium_mg": ("HKQuantityTypeIdentifierDietaryCalcium", "g"),
    "iron_mg": ("HKQuantityTypeIdentifierDietaryIron", "g"),
    "potassium_mg": ("HKQuantityTypeIdentifierDietaryPotassium", "g"),
    "vitamin_c_mg": ("HKQuantityTypeIdentifierDietaryVitaminC", "g"),
    "vitamin_d_mcg": ("HKQuantityTypeIdentifierDietaryVitaminD", "g"),
    "vitamin_b12_mcg": ("HKQuantityTypeIdentifierDietaryVitaminB12", "g"),
}


def _convert_to_hk_value(canonical_key: str, value: float) -> float:
    if canonical_key.endswith("_mg"):
        return value / 1000.0
    if canonical_key.endswith("_mcg"):
        return value / 1_000_000.0
    return value


def _normalize_micronutrient_value(unit: str, value: float, hk_unit: str) -> Optional[float]:
    low = unit.strip().lower()
    if hk_unit == "g":
        if low in ("g", "gram", "grams"):
            return value
        if low in ("mg", "milligram", "milligrams"):
            return value / 1000.0
        if low in ("mcg", "μg", "ug", "microgram", "micrograms"):
            return value / 1_000_000.0
        return None
    if hk_unit == "mL":
        return value if low in ("ml", "milliliter", "milliliters") else None
    if hk_unit == "kcal":
        return value if low in ("kcal", "kilocalorie", "kilocalories") else None
    return None


def nutrition_totals_to_apple_health_samples(totals: NutritionTotals) -> List[AppleHealthQuantitySample]:
    samples: List[AppleHealthQuantitySample] = []

    for canonical_key, (hk_identifier, hk_unit) in _CANONICAL_TO_HK.items():
        raw_value = getattr(totals, canonical_key)
        if raw_value is None or raw_value <= 0:
            continue
        samples.append(
            AppleHealthQuantitySample(
                identifier=hk_identifier,
                value=_convert_to_hk_value(canonical_key, float(raw_value)),
                unit=hk_unit,
            )
        )

    for key, nutrient in totals.micronutrients.items():
        mapping = _MICRONUTRIENT_ALIAS_TO_HK.get(key)
        if mapping:
            hk_identifier, hk_unit = mapping
            converted = _normalize_micronutrient_value(nutrient.unit, nutrient.value, hk_unit)
            if converted is None or converted <= 0:
                continue
            samples.append(AppleHealthQuantitySample(identifier=hk_identifier, value=converted, unit=hk_unit))
            continue

        # Advanced connector mode: allow direct HK identifier as key.
        if key.startswith("HKQuantityTypeIdentifierDietary"):
            converted = _normalize_micronutrient_value(nutrient.unit, nutrient.value, "g")
            if converted is None or converted <= 0:
                continue
            samples.append(AppleHealthQuantitySample(identifier=key, value=converted, unit="g"))

    return samples


def meal_log_to_apple_health_payload(meal_log: CanonicalMealLog) -> dict:
    """
    Build a connector-facing payload for iOS HealthKit integration.
    This is not direct HealthKit write code; it is an app-layer transfer payload.
    """
    return {
        "consumed_at": meal_log.consumed_at.isoformat(),
        "timezone": meal_log.timezone,
        "meal_type": meal_log.meal_type.value,
        "food_type": (meal_log.raw_inputs.text if meal_log.raw_inputs and meal_log.raw_inputs.text else None),
        "sync_identifier": f"meal:{meal_log.meal_id}",
        "sync_version": meal_log.record_version,
        "samples": [
            {"identifier": s.identifier, "value": s.value, "unit": s.unit}
            for s in nutrition_totals_to_apple_health_samples(meal_log.nutrition_totals)
        ],
    }
