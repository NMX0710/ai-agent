from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.nutrition.apple_health_mapping import meal_log_to_apple_health_payload
from app.nutrition.draft_store import MealDraftRecord, get_meal_draft_store
from app.nutrition.meallog import (
    CanonicalMealLog,
    ConfidenceInfo,
    ExternalSyncTarget,
    InputSource,
    MealLogStatus,
    MealType,
    NutritionTotals,
    RawInputs,
)
from app.tools.nutrition_http_tools import usda_search_foods


def _coerce_meal_type(value: Optional[str]) -> MealType:
    if not value:
        return MealType.unknown
    low = value.strip().lower()
    for item in MealType:
        if item.value == low:
            return item
    return MealType.unknown


def _estimate_nutrition_from_text(meal_description: str) -> tuple[NutritionTotals, float]:
    meal_text = (meal_description or "").strip()
    if not meal_text:
        return NutritionTotals(energy_kcal=0, protein_g=0, carbs_g=0, fat_g=0), 0.2

    data = usda_search_foods(query=meal_text, page_size=1, page_number=1)
    foods = data.get("foods") if isinstance(data, dict) else None
    if not isinstance(foods, list) or not foods:
        return NutritionTotals(energy_kcal=0, protein_g=0, carbs_g=0, fat_g=0), 0.3

    top = foods[0] if isinstance(foods[0], dict) else {}
    calories = top.get("calories_kcal")
    protein = top.get("protein_g")
    carbs = top.get("carbs_g")
    fat = top.get("fat_g")
    if not all(isinstance(v, (int, float)) for v in [calories, protein, carbs, fat]):
        return NutritionTotals(energy_kcal=0, protein_g=0, carbs_g=0, fat_g=0), 0.35

    return (
        NutritionTotals(
            energy_kcal=float(calories),
            protein_g=float(protein),
            carbs_g=float(carbs),
            fat_g=float(fat),
        ),
        0.65,
    )


def prepare_meal_log_draft(
    *,
    user_id: str,
    chat_id: str,
    input_source: InputSource,
    meal_description: str,
    consumed_at_iso: Optional[str] = None,
    timezone_name: str = "America/New_York",
    meal_type: Optional[str] = None,
    raw_channel: str = "telegram",
    raw_message_id: Optional[str] = None,
    image_ref: Optional[str] = None,
) -> dict[str, Any]:
    consumed_at = datetime.now(timezone.utc)
    if consumed_at_iso:
        try:
            consumed_at = datetime.fromisoformat(consumed_at_iso)
        except ValueError:
            consumed_at = datetime.now(timezone.utc)
    if consumed_at.tzinfo is None or consumed_at.tzinfo.utcoffset(consumed_at) is None:
        consumed_at = consumed_at.replace(tzinfo=timezone.utc)

    nutrition_totals, nutrition_confidence = _estimate_nutrition_from_text(meal_description)
    draft_id = uuid4().hex
    meal_log = CanonicalMealLog(
        meal_id=uuid4(),
        user_id=user_id,
        consumed_at=consumed_at,
        timezone=timezone_name,
        meal_type=_coerce_meal_type(meal_type),
        input_source=input_source,
        status=MealLogStatus.draft,
        nutrition_totals=nutrition_totals,
        confidence=ConfidenceInfo(
            overall=max(0.3, nutrition_confidence),
            nutrition=nutrition_confidence,
            item_recognition=0.35 if input_source == InputSource.image else None,
        ),
        raw_inputs=RawInputs(
            channel=raw_channel,
            message_id=raw_message_id,
            text=meal_description or None,
            image_refs=[image_ref] if image_ref else [],
        ),
        external_sync={
            "apple_health": ExternalSyncTarget(
                sync_status="pending",
                sync_identifier=f"meal:{draft_id}",
                sync_version=1,
            )
        },
    )
    now = datetime.now(timezone.utc)
    record = MealDraftRecord(
        draft_id=draft_id,
        user_id=user_id,
        chat_id=chat_id,
        meal_log=meal_log,
        status=MealLogStatus.draft.value,
        created_at=now,
        updated_at=now,
    )
    get_meal_draft_store().save(record)
    return {
        "ok": True,
        "draft_id": draft_id,
        "status": "draft",
        "nutrition_totals": meal_log.nutrition_totals.model_dump(),
        "meal_type": meal_log.meal_type.value,
    }


def commit_meal_log_draft(*, draft_id: str, user_id: str, confirmed: bool) -> dict[str, Any]:
    store = get_meal_draft_store()
    record = store.get(draft_id)
    if not record:
        return {"ok": False, "error": "draft_not_found"}
    if record.user_id != user_id:
        return {"ok": False, "error": "draft_user_mismatch"}
    if not confirmed:
        return {"ok": False, "error": "confirmation_required"}
    if record.status == MealLogStatus.synced.value:
        return {"ok": True, "status": "already_synced", "draft_id": draft_id}

    # Placeholder write result. The iOS companion should execute real HealthKit writes.
    payload = meal_log_to_apple_health_payload(record.meal_log)
    write_result = {
        "provider": "apple_health",
        "status": "synced",
        "external_id": f"apple-health:{draft_id}",
        "payload_preview": payload,
    }
    store.update(
        draft_id,
        status=MealLogStatus.synced.value,
        write_result=write_result,
        last_error=None,
    )
    return {"ok": True, "status": "synced", "draft_id": draft_id, "write_result": write_result}


def cancel_meal_log_draft(*, draft_id: str, user_id: str) -> dict[str, Any]:
    store = get_meal_draft_store()
    record = store.get(draft_id)
    if not record:
        return {"ok": False, "error": "draft_not_found"}
    if record.user_id != user_id:
        return {"ok": False, "error": "draft_user_mismatch"}
    if record.status == "cancelled":
        return {"ok": True, "status": "cancelled", "draft_id": draft_id}

    store.update(draft_id, status="cancelled")
    return {"ok": True, "status": "cancelled", "draft_id": draft_id}


def latest_pending_draft_for_chat(user_id: str, chat_id: str) -> Optional[MealDraftRecord]:
    return get_meal_draft_store().latest_pending_for_chat(user_id=user_id, chat_id=chat_id)


def mark_confirmation_prompted(draft_id: str) -> None:
    get_meal_draft_store().update(draft_id, confirmation_prompted=True)
