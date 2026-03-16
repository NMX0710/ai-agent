from __future__ import annotations

from datetime import datetime, timezone
import logging
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


def _coerce_meal_type(value: Optional[str]) -> MealType:
    if not value:
        return MealType.unknown
    low = value.strip().lower()
    for item in MealType:
        if item.value == low:
            return item
    return MealType.unknown


def _coerce_nutrition_totals(value: NutritionTotals | dict[str, Any]) -> NutritionTotals:
    if isinstance(value, NutritionTotals):
        return value
    if isinstance(value, dict):
        return NutritionTotals(**value)
    raise TypeError("nutrition_totals must be a NutritionTotals instance or dict")


def _coerce_nutrition_confidence(value: float | None) -> float:
    if value is None:
        return 0.6
    return min(1.0, max(0.0, float(value)))


def prepare_meal_log_draft(
    *,
    user_id: str,
    chat_id: str,
    input_source: InputSource,
    meal_description: str,
    nutrition_totals: NutritionTotals | dict[str, Any],
    nutrition_source: str,
    nutrition_confidence: float | None = None,
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

    normalized_description = " ".join((meal_description or "").strip().split())
    final_nutrition_totals = _coerce_nutrition_totals(nutrition_totals)
    final_nutrition_source = " ".join((nutrition_source or "").strip().split())
    final_nutrition_confidence = _coerce_nutrition_confidence(nutrition_confidence)
    if not final_nutrition_source:
        raise ValueError("nutrition_source is required")
    logging.info(
        "[MealDraft] final_estimate meal_description=%r source=%r calories=%s protein=%s carbs=%s fat=%s confidence=%s",
        normalized_description,
        final_nutrition_source,
        final_nutrition_totals.energy_kcal,
        final_nutrition_totals.protein_g,
        final_nutrition_totals.carbs_g,
        final_nutrition_totals.fat_g,
        final_nutrition_confidence,
    )
    draft_id = uuid4().hex
    meal_log = CanonicalMealLog(
        meal_id=uuid4(),
        user_id=user_id,
        consumed_at=consumed_at,
        timezone=timezone_name,
        meal_type=_coerce_meal_type(meal_type),
        input_source=input_source,
        status=MealLogStatus.draft,
        nutrition_totals=final_nutrition_totals,
        confidence=ConfidenceInfo(
            overall=max(0.3, final_nutrition_confidence),
            nutrition=final_nutrition_confidence,
            item_recognition=0.35 if input_source == InputSource.image else None,
        ),
        raw_inputs=RawInputs(
            channel=raw_channel,
            message_id=raw_message_id,
            text=normalized_description or None,
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
    logging.info(
        "[MealDraft] created draft_id=%s user_id=%s chat_id=%s calories=%s protein=%s carbs=%s fat=%s confidence=%s",
        draft_id,
        user_id,
        chat_id,
        meal_log.nutrition_totals.energy_kcal,
        meal_log.nutrition_totals.protein_g,
        meal_log.nutrition_totals.carbs_g,
        meal_log.nutrition_totals.fat_g,
        final_nutrition_confidence,
    )
    return {
        "ok": True,
        "draft_id": draft_id,
        "status": "draft",
        "nutrition_totals": meal_log.nutrition_totals.model_dump(),
        "meal_type": meal_log.meal_type.value,
        "nutrition_source": final_nutrition_source,
        "nutrition_confidence": final_nutrition_confidence,
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
    if record.status == "cancelled":
        return {"ok": False, "error": "draft_cancelled"}

    payload = meal_log_to_apple_health_payload(record.meal_log)
    store.update(
        draft_id,
        status=MealLogStatus.confirmed.value,
        bridge_dispatched=False,
        bridge_claim_token=None,
        bridge_claimed_at=None,
        bridge_claim_expires_at=None,
        write_result=None,
        last_error=None,
    )
    return {
        "ok": True,
        "status": "pending_device_write",
        "draft_id": draft_id,
        "bridge_payload": payload,
    }


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


def list_pending_apple_health_writes(
    *,
    user_id: str,
    limit: int = 20,
    lease_seconds: int = 300,
) -> list[dict[str, Any]]:
    rows = get_meal_draft_store().claim_for_user(
        user_id=user_id,
        status=MealLogStatus.confirmed.value,
        limit=limit,
        lease_seconds=lease_seconds,
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "draft_id": row.draft_id,
                "user_id": row.user_id,
                "chat_id": row.chat_id,
                "status": row.status,
                "bridge_dispatched": row.bridge_dispatched,
                "bridge_attempt_count": row.bridge_attempt_count,
                "claim_token": row.bridge_claim_token,
                "claim_expires_at": row.bridge_claim_expires_at.isoformat() if row.bridge_claim_expires_at else None,
                "payload": meal_log_to_apple_health_payload(row.meal_log),
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
        )
    return out


def report_apple_health_write_result(
    *,
    draft_id: str,
    user_id: str,
    success: bool,
    claim_token: Optional[str] = None,
    external_id: Optional[str] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    store = get_meal_draft_store()
    record = store.get(draft_id)
    if not record:
        return {"ok": False, "error": "draft_not_found"}
    if record.user_id != user_id:
        return {"ok": False, "error": "draft_user_mismatch"}
    if record.status == "cancelled":
        return {"ok": False, "error": "draft_cancelled"}
    if record.status == MealLogStatus.synced.value:
        return {
            "ok": True,
            "status": "synced",
            "draft_id": draft_id,
            "write_result": record.write_result,
        }

    active_claim_token = record.bridge_claim_token
    claim_expired = bool(
        record.bridge_claim_expires_at
        and record.bridge_claim_expires_at <= datetime.now(timezone.utc)
    )
    effective_claim_token = claim_token or active_claim_token
    if record.last_claim_token and claim_token and claim_token == record.last_claim_token and record.status == MealLogStatus.sync_failed.value:
        return {
            "ok": True,
            "status": MealLogStatus.sync_failed.value,
            "draft_id": draft_id,
            "error": record.last_error or "unknown_error",
        }
    if not effective_claim_token:
        return {"ok": False, "error": "claim_token_required"}
    if not active_claim_token:
        return {"ok": False, "error": "draft_not_claimed"}
    if claim_expired:
        return {"ok": False, "error": "claim_expired"}
    if effective_claim_token != active_claim_token:
        return {"ok": False, "error": "claim_token_mismatch"}

    if success:
        write_result = {
            "provider": "apple_health",
            "status": "synced",
            "external_id": external_id or f"apple-health:{draft_id}",
        }
        store.update(
            draft_id,
            status=MealLogStatus.synced.value,
            bridge_claim_token=None,
            bridge_claimed_at=None,
            bridge_claim_expires_at=None,
            write_result=write_result,
            last_error=None,
        )
        return {"ok": True, "status": "synced", "draft_id": draft_id, "write_result": write_result}

    store.update(
        draft_id,
        status=MealLogStatus.sync_failed.value,
        bridge_claim_token=None,
        bridge_claimed_at=None,
        bridge_claim_expires_at=None,
        write_result=None,
        last_error=error or "unknown_error",
    )
    return {
        "ok": True,
        "status": MealLogStatus.sync_failed.value,
        "draft_id": draft_id,
        "error": error or "unknown_error",
    }
