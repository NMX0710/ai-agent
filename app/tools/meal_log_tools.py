from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from app.nutrition.meal_log_service import commit_meal_log_draft, prepare_meal_log_draft
from app.nutrition.meallog import InputSource


@tool(
    description=(
        "Prepare a canonical meal log draft from user meal input. "
        "Use this when the user asks to record/log a meal. "
        "meal_description is user-facing meal text in the user's language. "
        "food_query must be a clean food phrase only, with no wrappers like 'I ate', '帮我记录', or time-of-day filler. "
        "If the original input is Chinese or mixed-language, also pass food_query_en as the English lookup phrase for USDA/Spoonacular."
    )
)
def prepare_meal_log(
    user_id: str,
    chat_id: str,
    meal_description: str,
    food_query: str | None = None,
    food_query_en: str | None = None,
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
        "[MealLogTool] prepare_meal_log meal_description=%r food_query=%r food_query_en=%r",
        meal_description,
        food_query,
        food_query_en,
    )
    return prepare_meal_log_draft(
        user_id=user_id,
        chat_id=chat_id,
        input_source=source,
        meal_description=meal_description,
        food_query=food_query,
        food_query_en=food_query_en,
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
