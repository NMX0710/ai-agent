from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from app.nutrition.meal_log_service import commit_meal_log_draft, prepare_meal_log_draft
from app.nutrition.meallog import InputSource, NutritionTotals


@tool(
    description=(
        "Prepare a canonical meal log draft from a final nutrition estimate that the agent has already chosen. "
        "Use this only after the agent has used nutrition lookup tools and selected one final kcal/protein/carbs/fat estimate. "
        "Do not call this tool with placeholder sources, all-zero macros, or unknown final values. "
        "If you do not yet have a usable final estimate, ask a clarification question or explicitly choose an approximate estimate first. "
        "This tool does not perform nutrition lookup."
    )
)
def prepare_meal_log(
    user_id: str,
    chat_id: str,
    meal_description: str,
    energy_kcal: float,
    protein_g: float,
    carbs_g: float,
    fat_g: float,
    nutrition_source: str,
    nutrition_confidence: float | None = None,
    consumed_at_iso: str | None = None,
    timezone_name: str = "America/New_York",
    meal_type: str | None = None,
    input_source: str = "text",
    raw_channel: str = "telegram",
    raw_message_id: str | None = None,
    image_ref: str | None = None,
) -> dict[str, Any]:
    source = InputSource(input_source) if input_source in InputSource._value2member_map_ else InputSource.text
    logging.info(
        "[MealLogTool] prepare_meal_log meal_description=%r source=%r calories=%s protein=%s carbs=%s fat=%s",
        meal_description,
        nutrition_source,
        energy_kcal,
        protein_g,
        carbs_g,
        fat_g,
    )
    return prepare_meal_log_draft(
        user_id=user_id,
        chat_id=chat_id,
        input_source=source,
        meal_description=meal_description,
        nutrition_totals=NutritionTotals(
            energy_kcal=energy_kcal,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
        ),
        nutrition_source=nutrition_source,
        nutrition_confidence=nutrition_confidence,
        consumed_at_iso=consumed_at_iso,
        timezone_name=timezone_name,
        meal_type=meal_type,
        raw_channel=raw_channel,
        raw_message_id=raw_message_id,
        image_ref=image_ref,
    )


@tool(
    description=(
        "Commit a prepared meal log draft into Apple Health flow. "
        "Only call this after explicit user confirmation."
    )
)
def commit_meal_log(
    draft_id: str,
    user_id: str,
    confirmed: bool,
) -> dict[str, Any]:
    return commit_meal_log_draft(draft_id=draft_id, user_id=user_id, confirmed=confirmed)
