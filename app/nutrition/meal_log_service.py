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
from app.nutrition.query_normalization import NormalizedMealQuery, normalize_meal_query
from app.tools.nutrition_http_tools import (
    openfoodfacts_search_products,
    spoonacular_search_recipe,
    tavily_search_nutrition,
    usda_search_foods,
)


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


def _select_lookup_queries(query: NormalizedMealQuery) -> list[str]:
    candidates: list[str] = []
    for item in [query.food_query_en, query.food_query]:
        value = " ".join((item or "").strip().split())
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def _totals_from_macros(calories: Any, protein: Any, carbs: Any, fat: Any) -> NutritionTotals:
    return NutritionTotals(
        energy_kcal=float(calories),
        protein_g=float(protein),
        carbs_g=float(carbs),
        fat_g=float(fat),
    )


def _extract_tavily_macros(text: str) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    low = text.lower()
    patterns = {
        "calories": r"(\d+(?:\.\d+)?)\s*(?:kcal|calories|calorie)",
        "protein": r"protein[^0-9]{0,20}(\d+(?:\.\d+)?)\s*g|\b(\d+(?:\.\d+)?)\s*g[^.\n]{0,20}protein",
        "carbs": r"carb(?:s|ohydrates)?[^0-9]{0,20}(\d+(?:\.\d+)?)\s*g|\b(\d+(?:\.\d+)?)\s*g[^.\n]{0,20}carb",
        "fat": r"fat[^0-9]{0,20}(\d+(?:\.\d+)?)\s*g|\b(\d+(?:\.\d+)?)\s*g[^.\n]{0,20}fat",
    }
    values: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, low)
        if not match:
            return None
        for group in match.groups():
            if group is not None:
                values[key] = float(group)
                break
    if len(values) != 4:
        return None
    return values["calories"], values["protein"], values["carbs"], values["fat"]


def _try_usda_lookup(query: NormalizedMealQuery, lookup_queries: list[str]) -> tuple[NutritionTotals, float, str] | None:
    usda_has_complete_macros = False
    for lookup_query in lookup_queries:
        data = usda_search_foods.invoke({"query": lookup_query, "page_size": 5, "page_number": 1})
        foods = data.get("foods") if isinstance(data, dict) else None
        usda_count = len(foods) if isinstance(foods, list) else 0
        logging.info(
            "[MealEstimate] source=usda stage=query raw_query=%r food_query=%r food_query_en=%r lookup_query=%r hits=%s",
            query.user_text,
            query.food_query,
            query.food_query_en,
            lookup_query,
            usda_count,
        )
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
                    return _totals_from_macros(calories, protein, carbs, fat), 0.68, "USDA"
            if usda_count == 0:
                logging.info("[MealEstimate] source=usda stage=selected reason=no_hits")
            elif not usda_has_complete_macros:
                logging.info("[MealEstimate] source=usda stage=selected reason=has_hits_but_missing_macros")
    return None


def _try_spoonacular_lookup(query: NormalizedMealQuery, lookup_queries: list[str]) -> tuple[NutritionTotals, float, str] | None:
    spoon_has_complete_macros = False
    for lookup_query in lookup_queries:
        recipe_data = spoonacular_search_recipe.invoke({"query": lookup_query, "number": 5})
        recipes = recipe_data.get("results") if isinstance(recipe_data, dict) else None
        spoon_count = len(recipes) if isinstance(recipes, list) else 0
        logging.info(
            "[MealEstimate] source=spoonacular stage=query raw_query=%r food_query=%r food_query_en=%r lookup_query=%r hits=%s",
            query.user_text,
            query.food_query,
            query.food_query_en,
            lookup_query,
            spoon_count,
        )
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
                    return _totals_from_macros(calories, protein, carbs, fat), 0.6, "Spoonacular"
            if spoon_count == 0:
                logging.info("[MealEstimate] source=spoonacular stage=selected reason=no_hits")
            elif not spoon_has_complete_macros:
                logging.info("[MealEstimate] source=spoonacular stage=selected reason=has_hits_but_missing_macros")
    return None


def _try_tavily_lookup(query: NormalizedMealQuery, lookup_queries: list[str]) -> tuple[NutritionTotals, float, str] | None:
    for lookup_query in lookup_queries:
        data = tavily_search_nutrition.invoke({"query": lookup_query, "max_results": 5})
        results = data.get("results") if isinstance(data, dict) else None
        tavily_count = len(results) if isinstance(results, list) else 0
        logging.info(
            "[MealEstimate] source=tavily stage=query raw_query=%r food_query=%r food_query_en=%r lookup_query=%r hits=%s",
            query.user_text,
            query.food_query,
            query.food_query_en,
            lookup_query,
            tavily_count,
        )
        if not isinstance(results, list):
            continue
        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            content = str(item.get("content") or "")
            parsed = _extract_tavily_macros(content)
            logging.info(
                "[MealEstimate] source=tavily stage=candidate idx=%s title=%r parsed=%r",
                idx,
                title,
                parsed,
            )
            if not parsed:
                continue
            calories, protein, carbs, fat = parsed
            logging.info(
                "[MealEstimate] source=tavily stage=selected idx=%s title=%r calories=%s protein=%s carbs=%s fat=%s",
                idx,
                title,
                calories,
                protein,
                carbs,
                fat,
            )
            return _totals_from_macros(calories, protein, carbs, fat), 0.5, "Tavily"
        if tavily_count == 0:
            logging.info("[MealEstimate] source=tavily stage=selected reason=no_hits")
    return None


def _try_openfoodfacts_lookup(query: NormalizedMealQuery, lookup_queries: list[str]) -> tuple[NutritionTotals, float, str] | None:
    for lookup_query in lookup_queries:
        data = openfoodfacts_search_products.invoke({"query": lookup_query, "page_size": 5})
        products = data.get("products") if isinstance(data, dict) else None
        off_count = len(products) if isinstance(products, list) else 0
        logging.info(
            "[MealEstimate] source=openfoodfacts stage=query raw_query=%r food_query=%r food_query_en=%r lookup_query=%r hits=%s",
            query.user_text,
            query.food_query,
            query.food_query_en,
            lookup_query,
            off_count,
        )
        if not isinstance(products, list):
            continue
        for idx, item in enumerate(products):
            if not isinstance(item, dict):
                continue
            product_name = item.get("product_name")
            calories = item.get("calories_kcal")
            protein = item.get("protein_g")
            carbs = item.get("carbs_g")
            fat = item.get("fat_g")
            logging.info(
                "[MealEstimate] source=openfoodfacts stage=candidate idx=%s product_name=%r calories=%r protein=%r carbs=%r fat=%r",
                idx,
                product_name,
                calories,
                protein,
                carbs,
                fat,
            )
            if _has_complete_macros(calories, protein, carbs, fat):
                logging.info(
                    "[MealEstimate] source=openfoodfacts stage=selected idx=%s product_name=%r calories=%s protein=%s carbs=%s fat=%s",
                    idx,
                    product_name,
                    float(calories),
                    float(protein),
                    float(carbs),
                    float(fat),
                )
                return _totals_from_macros(calories, protein, carbs, fat), 0.58, "OpenFoodFacts"
        if off_count == 0:
            logging.info("[MealEstimate] source=openfoodfacts stage=selected reason=no_hits")
    return None


def _estimate_nutrition_from_text(query: NormalizedMealQuery) -> tuple[NutritionTotals, float, str]:
    meal_text = query.meal_description
    lookup_queries = _select_lookup_queries(query)
    lookup_query = lookup_queries[0] if lookup_queries else ""
    if not meal_text:
        logging.info(
            "[MealEstimate] source=none reason=empty_input raw_query=%r lookup_query=%r",
            query.user_text,
            lookup_query,
        )
        return NutritionTotals(energy_kcal=0, protein_g=0, carbs_g=0, fat_g=0), 0.2, "Estimated"

    for lookup_fn in [
        _try_usda_lookup,
        _try_spoonacular_lookup,
        _try_tavily_lookup,
        _try_openfoodfacts_lookup,
    ]:
        result = lookup_fn(query, lookup_queries)
        if result:
            return result

    logging.info(
        "[MealEstimate] source=none reason=all_sources_missing_macros raw_query=%r food_query=%r food_query_en=%r",
        query.user_text,
        query.food_query,
        query.food_query_en,
    )
    llm_fallback = _infer_nutrition_with_llm(query.food_query_en or query.food_query or meal_text)
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
        return totals, confidence, "Estimated"

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
    return hard_fallback, 0.25, "Estimated"


def prepare_meal_log_draft(
    *,
    user_id: str,
    chat_id: str,
    input_source: InputSource,
    meal_description: str,
    food_query: Optional[str] = None,
    food_query_en: Optional[str] = None,
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

    normalized_query = normalize_meal_query(
        user_text=meal_description,
        meal_description=meal_description,
        food_query=food_query,
        food_query_en=food_query_en,
    )
    logging.info(
        "[MealDraft] normalized_query meal_description=%r food_query=%r food_query_en=%r detected_language=%s",
        normalized_query.meal_description,
        normalized_query.food_query,
        normalized_query.food_query_en,
        normalized_query.detected_language,
    )
    nutrition_totals, nutrition_confidence, nutrition_source = _estimate_nutrition_from_text(normalized_query)
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
        "food_query": normalized_query.food_query,
        "food_query_en": normalized_query.food_query_en,
        "nutrition_source": nutrition_source,
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
