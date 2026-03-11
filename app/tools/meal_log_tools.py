from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.nutrition.meal_log_service import commit_meal_log_draft, prepare_meal_log_draft
from app.nutrition.meallog import InputSource


@tool(
    description=(
        "Prepare a canonical meal log draft from user meal input. "
        "Use this when the user asks to record/log a meal."
    )
)
def prepare_meal_log(
    user_id: str,
    chat_id: str,
    meal_description: str,
    consumed_at_iso: str | None = None,
    timezone_name: str = "America/New_York",
    meal_type: str | None = None,
    input_source: str = "text",
    raw_channel: str = "telegram",
    raw_message_id: str | None = None,
    image_ref: str | None = None,
) -> dict[str, Any]:
    source = InputSource(input_source) if input_source in InputSource._value2member_map_ else InputSource.text
    return prepare_meal_log_draft(
        user_id=user_id,
        chat_id=chat_id,
        input_source=source,
        meal_description=meal_description,
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
