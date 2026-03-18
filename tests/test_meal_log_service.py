import sys
from pathlib import Path

import pytest

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.nutrition.draft_store import get_meal_draft_store
from app.nutrition.meal_log_service import (
    commit_meal_log_draft,
    list_pending_apple_health_writes,
    prepare_meal_log_draft,
    report_apple_health_write_result,
)
from app.nutrition.meallog import InputSource, NutritionTotals


def _reset_store():
    store = get_meal_draft_store()
    store._records.clear()  # test-only reset of singleton in-memory store


def _sample_totals() -> NutritionTotals:
    return NutritionTotals(energy_kcal=500, protein_g=30, carbs_g=40, fat_g=20)


def test_prepare_uses_agent_selected_estimate_and_default_confidence():
    _reset_store()

    draft = prepare_meal_log_draft(
        user_id="tg:1",
        chat_id="tg:chat1",
        input_source=InputSource.text,
        meal_description="rice and chicken",
        nutrition_totals=_sample_totals(),
        nutrition_source="USDA",
    )

    assert draft["ok"] is True
    assert draft["nutrition_totals"] == _sample_totals().model_dump()
    assert draft["nutrition_source"] == "USDA"
    assert draft["nutrition_confidence"] == 0.6


def test_prepare_accepts_dict_nutrition_totals_and_explicit_confidence():
    _reset_store()

    draft = prepare_meal_log_draft(
        user_id="tg:dict",
        chat_id="tg:chat-dict",
        input_source=InputSource.text,
        meal_description="protein bar",
        nutrition_totals={
            "energy_kcal": 210,
            "protein_g": 20,
            "carbs_g": 18,
            "fat_g": 7,
        },
        nutrition_source="OpenFoodFacts",
        nutrition_confidence=0.82,
    )

    assert draft["nutrition_totals"]["energy_kcal"] == 210
    assert draft["nutrition_source"] == "OpenFoodFacts"
    assert draft["nutrition_confidence"] == 0.82


def test_prepare_requires_nutrition_source():
    _reset_store()

    with pytest.raises(ValueError, match="nutrition_source is required"):
        prepare_meal_log_draft(
            user_id="tg:missing-source",
            chat_id="tg:chat-missing-source",
            input_source=InputSource.text,
            meal_description="toast",
            nutrition_totals=_sample_totals(),
            nutrition_source="   ",
        )


def test_prepare_rejects_placeholder_nutrition_source():
    _reset_store()

    with pytest.raises(ValueError, match="nutrition_source must identify a real nutrition estimate source"):
        prepare_meal_log_draft(
            user_id="tg:placeholder-source",
            chat_id="tg:chat-placeholder-source",
            input_source=InputSource.text,
            meal_description="spaghetti bolognese",
            nutrition_totals=_sample_totals(),
            nutrition_source="user_input",
        )


def test_prepare_rejects_all_zero_final_estimate():
    _reset_store()

    with pytest.raises(ValueError, match="final nutrition estimate cannot be all zeros"):
        prepare_meal_log_draft(
            user_id="tg:zero-estimate",
            chat_id="tg:chat-zero-estimate",
            input_source=InputSource.text,
            meal_description="spaghetti bolognese",
            nutrition_totals=NutritionTotals(
                energy_kcal=0,
                protein_g=0,
                carbs_g=0,
                fat_g=0,
            ),
            nutrition_source="Estimated",
        )


def test_commit_generates_pending_device_write_then_report_synced():
    _reset_store()

    draft = prepare_meal_log_draft(
        user_id="tg:2",
        chat_id="tg:chat2",
        input_source=InputSource.text,
        meal_description="rice and chicken",
        nutrition_totals=_sample_totals(),
        nutrition_source="USDA",
        nutrition_confidence=0.7,
    )
    draft_id = draft["draft_id"]

    commit = commit_meal_log_draft(draft_id=draft_id, user_id="tg:2", confirmed=True)
    assert commit["ok"] is True
    assert commit["status"] == "pending_device_write"
    assert isinstance(commit["bridge_payload"], dict)
    assert commit["bridge_payload"]["samples"]

    pending = list_pending_apple_health_writes(user_id="tg:2")
    assert len(pending) == 1
    assert pending[0]["draft_id"] == draft_id

    synced = report_apple_health_write_result(
        draft_id=draft_id,
        user_id="tg:2",
        success=True,
        external_id="hk-abc",
    )
    assert synced["ok"] is True
    assert synced["status"] == "synced"
    assert synced["write_result"]["external_id"] == "hk-abc"

    pending_after = list_pending_apple_health_writes(user_id="tg:2")
    assert pending_after == []


def test_report_failure_marks_sync_failed():
    _reset_store()

    draft = prepare_meal_log_draft(
        user_id="tg:3",
        chat_id="tg:chat3",
        input_source=InputSource.text,
        meal_description="toast",
        nutrition_totals=NutritionTotals(energy_kcal=300, protein_g=10, carbs_g=20, fat_g=15),
        nutrition_source="Estimated",
        nutrition_confidence=0.55,
    )
    draft_id = draft["draft_id"]

    commit = commit_meal_log_draft(draft_id=draft_id, user_id="tg:3", confirmed=True)
    assert commit["status"] == "pending_device_write"
    claimed = list_pending_apple_health_writes(user_id="tg:3")
    assert len(claimed) == 1

    failed = report_apple_health_write_result(
        draft_id=draft_id,
        user_id="tg:3",
        success=False,
        claim_token=claimed[0]["claim_token"],
        error="bridge_timeout",
    )
    assert failed["ok"] is True
    assert failed["status"] == "sync_failed"
    assert failed["error"] == "bridge_timeout"
