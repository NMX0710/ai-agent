from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from langchain_openai import ChatOpenAI

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
from app.tools.nutrition_http_tools import spoonacular_search_recipe, usda_search_foods


def _normalize_lookup_query(text: str) -> str:
    return " ".join((text or "").strip().split())


def _has_complete_macros(calories: Any, protein: Any, carbs: Any, fat: Any) -> bool:
    return all(isinstance(v, (int, float)) for v in [calories, protein, carbs, fat])


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _infer_nutrition_with_llm(meal_text: str) -> tuple[NutritionTotals, float] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    prompt = (
        "Estimate nutrition for one typical serving of this meal.\n"
        f"Meal: {meal_text}\n\n"
        "Return JSON only with numeric fields:\n"
        '{"energy_kcal": number, "protein_g": number, "carbs_g": number, "fat_g": number}'
    )
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            content_text = "\n".join(chunks).strip()
        else:
            content_text = str(content or "").strip()
        parsed = _parse_json_object(content_text)
        if not parsed:
            return None

        calories = parsed.get("energy_kcal")
        protein = parsed.get("protein_g")
        carbs = parsed.get("carbs_g")
        fat = parsed.get("fat_g")
        if not _has_complete_macros(calories, protein, carbs, fat):
            return None

        return (
            NutritionTotals(
                energy_kcal=max(1.0, float(calories)),
                protein_g=max(0.1, float(protein)),
                carbs_g=max(0.1, float(carbs)),
                fat_g=max(0.1, float(fat)),
            ),
            0.45,
        )
    except Exception:
        return None


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
    lookup_query = _normalize_lookup_query(meal_text)
    if not meal_text:
        logging.info(
            "[MealEstimate] source=none reason=empty_input raw_query=%r lookup_query=%r",
            meal_description,
            lookup_query,
        )
        return NutritionTotals(energy_kcal=0, protein_g=0, carbs_g=0, fat_g=0), 0.2

    # usda_search_foods/spoonacular_search_recipe are LangChain StructuredTool (@tool), invoke with dict args.
    data = usda_search_foods.invoke({"query": lookup_query, "page_size": 5, "page_number": 1})
    foods = data.get("foods") if isinstance(data, dict) else None
    usda_count = len(foods) if isinstance(foods, list) else 0
    logging.info(
        "[MealEstimate] source=usda stage=query raw_query=%r lookup_query=%r hits=%s",
        meal_description,
        lookup_query,
        usda_count,
    )
    usda_has_complete_macros = False
    if isinstance(foods, list):
        for idx, item in enumerate(foods):
            if not isinstance(item, dict):
                continue
            desc = item.get("description")
            calories = item.get("calories_kcal")
            protein = item.get("protein_g")
            carbs = item.get("carbs_g")
            fat = item.get("fat_g")
            logging.info(
                "[MealEstimate] source=usda stage=candidate idx=%s description=%r calories=%r protein=%r carbs=%r fat=%r",
                idx,
                desc,
                calories,
                protein,
                carbs,
                fat,
            )
            if _has_complete_macros(calories, protein, carbs, fat):
                usda_has_complete_macros = True
                logging.info(
                    "[MealEstimate] source=usda stage=selected idx=%s description=%r calories=%s protein=%s carbs=%s fat=%s",
                    idx,
                    desc,
                    float(calories),
                    float(protein),
                    float(carbs),
                    float(fat),
                )
                return (
                    NutritionTotals(
                        energy_kcal=float(calories),
                        protein_g=float(protein),
                        carbs_g=float(carbs),
                        fat_g=float(fat),
                    ),
                    0.68,
                )
        if usda_count == 0:
            logging.info("[MealEstimate] source=usda stage=selected reason=no_hits")
        elif not usda_has_complete_macros:
            logging.info("[MealEstimate] source=usda stage=selected reason=has_hits_but_missing_macros")

    recipe_data = spoonacular_search_recipe.invoke({"query": lookup_query, "number": 5})
    recipes = recipe_data.get("results") if isinstance(recipe_data, dict) else None
    spoon_count = len(recipes) if isinstance(recipes, list) else 0
    logging.info(
        "[MealEstimate] source=spoonacular stage=query raw_query=%r lookup_query=%r hits=%s",
        meal_description,
        lookup_query,
        spoon_count,
    )
    spoon_has_complete_macros = False
    if isinstance(recipes, list):
        for idx, item in enumerate(recipes):
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            calories = item.get("calories")
            protein = item.get("protein_g")
            carbs = item.get("carbs_g")
            fat = item.get("fat_g")
            logging.info(
                "[MealEstimate] source=spoonacular stage=candidate idx=%s title=%r calories=%r protein=%r carbs=%r fat=%r",
                idx,
                title,
                calories,
                protein,
                carbs,
                fat,
            )
            if _has_complete_macros(calories, protein, carbs, fat):
                spoon_has_complete_macros = True
                logging.info(
                    "[MealEstimate] source=spoonacular stage=selected idx=%s title=%r calories=%s protein=%s carbs=%s fat=%s",
                    idx,
                    title,
                    float(calories),
                    float(protein),
                    float(carbs),
                    float(fat),
                )
                return (
                    NutritionTotals(
                        energy_kcal=float(calories),
                        protein_g=float(protein),
                        carbs_g=float(carbs),
                        fat_g=float(fat),
                    ),
                    0.6,
                )
        if spoon_count == 0:
            logging.info("[MealEstimate] source=spoonacular stage=selected reason=no_hits")
        elif not spoon_has_complete_macros:
            logging.info("[MealEstimate] source=spoonacular stage=selected reason=has_hits_but_missing_macros")

    logging.info(
        "[MealEstimate] source=none reason=all_sources_missing_macros raw_query=%r lookup_query=%r",
        meal_description,
        lookup_query,
    )
    llm_fallback = _infer_nutrition_with_llm(lookup_query)
    if llm_fallback:
        totals, confidence = llm_fallback
        logging.info(
            "[MealEstimate] source=llm_fallback stage=selected calories=%s protein=%s carbs=%s fat=%s confidence=%s",
            totals.energy_kcal,
            totals.protein_g,
            totals.carbs_g,
            totals.fat_g,
            confidence,
        )
        return totals, confidence

    # Hard fallback to avoid returning empty/zero record when all upstream estimates fail.
    hard_fallback = NutritionTotals(
        energy_kcal=450.0,
        protein_g=18.0,
        carbs_g=55.0,
        fat_g=15.0,
    )
    logging.info(
        "[MealEstimate] source=hard_fallback stage=selected calories=%s protein=%s carbs=%s fat=%s confidence=%s",
        hard_fallback.energy_kcal,
        hard_fallback.protein_g,
        hard_fallback.carbs_g,
        hard_fallback.fat_g,
        0.25,
    )
    return hard_fallback, 0.25


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
    logging.info(
        "[MealDraft] created draft_id=%s user_id=%s chat_id=%s calories=%s protein=%s carbs=%s fat=%s confidence=%s",
        draft_id,
        user_id,
        chat_id,
        meal_log.nutrition_totals.energy_kcal,
        meal_log.nutrition_totals.protein_g,
        meal_log.nutrition_totals.carbs_g,
        meal_log.nutrition_totals.fat_g,
        nutrition_confidence,
    )
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
    if record.status == "cancelled":
        return {"ok": False, "error": "draft_cancelled"}

    payload = meal_log_to_apple_health_payload(record.meal_log)
    store.update(
        draft_id,
        status=MealLogStatus.confirmed.value,
        bridge_dispatched=False,
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


def list_pending_apple_health_writes(*, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = get_meal_draft_store().list_for_user(user_id=user_id, status=MealLogStatus.confirmed.value, limit=limit)
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "draft_id": row.draft_id,
                "user_id": row.user_id,
                "chat_id": row.chat_id,
                "status": row.status,
                "bridge_dispatched": row.bridge_dispatched,
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

    if success:
        write_result = {
            "provider": "apple_health",
            "status": "synced",
            "external_id": external_id or f"apple-health:{draft_id}",
        }
        store.update(
            draft_id,
            status=MealLogStatus.synced.value,
            bridge_dispatched=True,
            write_result=write_result,
            last_error=None,
        )
        return {"ok": True, "status": "synced", "draft_id": draft_id, "write_result": write_result}

    store.update(
        draft_id,
        status=MealLogStatus.sync_failed.value,
        bridge_dispatched=True,
        write_result=None,
        last_error=error or "unknown_error",
    )
    return {
        "ok": True,
        "status": MealLogStatus.sync_failed.value,
        "draft_id": draft_id,
        "error": error or "unknown_error",
    }
