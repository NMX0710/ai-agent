import sys
from pathlib import Path

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


def test_commit_generates_pending_device_write_then_report_synced(monkeypatch):
    _reset_store()

    monkeypatch.setattr(
        "app.nutrition.meal_log_service._estimate_nutrition_from_text",
        lambda _: (NutritionTotals(energy_kcal=500, protein_g=30, carbs_g=40, fat_g=20), 0.7),
    )

    draft = prepare_meal_log_draft(
        user_id="tg:1",
        chat_id="tg:chat1",
        input_source=InputSource.text,
        meal_description="rice and chicken",
    )
    draft_id = draft["draft_id"]

    commit = commit_meal_log_draft(draft_id=draft_id, user_id="tg:1", confirmed=True)
    assert commit["ok"] is True
    assert commit["status"] == "pending_device_write"
    assert isinstance(commit["bridge_payload"], dict)
    assert commit["bridge_payload"]["samples"]

    pending = list_pending_apple_health_writes(user_id="tg:1")
    assert len(pending) == 1
    assert pending[0]["draft_id"] == draft_id

    synced = report_apple_health_write_result(
        draft_id=draft_id,
        user_id="tg:1",
        success=True,
        external_id="hk-abc",
    )
    assert synced["ok"] is True
    assert synced["status"] == "synced"
    assert synced["write_result"]["external_id"] == "hk-abc"

    pending_after = list_pending_apple_health_writes(user_id="tg:1")
    assert pending_after == []


def test_report_failure_marks_sync_failed(monkeypatch):
    _reset_store()

    monkeypatch.setattr(
        "app.nutrition.meal_log_service._estimate_nutrition_from_text",
        lambda _: (NutritionTotals(energy_kcal=300, protein_g=10, carbs_g=20, fat_g=15), 0.6),
    )

    draft = prepare_meal_log_draft(
        user_id="tg:2",
        chat_id="tg:chat2",
        input_source=InputSource.text,
        meal_description="toast",
    )
    draft_id = draft["draft_id"]

    commit = commit_meal_log_draft(draft_id=draft_id, user_id="tg:2", confirmed=True)
    assert commit["status"] == "pending_device_write"

    failed = report_apple_health_write_result(
        draft_id=draft_id,
        user_id="tg:2",
        success=False,
        error="bridge_timeout",
    )
    assert failed["ok"] is True
    assert failed["status"] == "sync_failed"
    assert failed["error"] == "bridge_timeout"


def test_prepare_uses_tool_invoke_contract(monkeypatch):
    _reset_store()

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            assert payload["query"] == "spaghetti bolognese"
            return {
                "foods": [
                    {
                        "calories_kcal": 510,
                        "protein_g": 23,
                        "carbs_g": 62,
                        "fat_g": 18,
                    }
                ]
            }

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())

    draft = prepare_meal_log_draft(
        user_id="tg:3",
        chat_id="tg:chat3",
        input_source=InputSource.text,
        meal_description="spaghetti bolognese",
    )
    assert draft["ok"] is True
    totals = draft["nutrition_totals"]
    assert totals["energy_kcal"] == 510
    assert totals["protein_g"] == 23


def test_estimate_scans_multiple_usda_results(monkeypatch):
    _reset_store()

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            return {
                "foods": [
                    {"calories_kcal": None, "protein_g": None, "carbs_g": None, "fat_g": None},
                    {"calories_kcal": 430, "protein_g": 18, "carbs_g": 55, "fat_g": 14},
                ]
            }

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            return {"results": []}

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())

    draft = prepare_meal_log_draft(
        user_id="tg:4",
        chat_id="tg:chat4",
        input_source=InputSource.text,
        meal_description="pasta",
    )
    totals = draft["nutrition_totals"]
    assert totals["energy_kcal"] == 430
    assert totals["protein_g"] == 18
    assert totals["carbs_g"] == 55
    assert totals["fat_g"] == 14


def test_prepare_passes_normalized_query_without_rule_based_extraction(monkeypatch):
    _reset_store()
    captured = {}

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            captured["query"] = payload["query"]
            return {
                "foods": [
                    {"calories_kcal": 520, "protein_g": 20, "carbs_g": 65, "fat_g": 18},
                ]
            }

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            return {"results": []}

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())

    prepare_meal_log_draft(
        user_id="tg:5",
        chat_id="tg:chat5",
        input_source=InputSource.text,
        meal_description="我晚上吃了意大利面 可以帮我记录吗",
    )
    assert captured["query"] == "我晚上吃了意大利面 可以帮我记录吗"


def test_prepare_uses_llm_fallback_when_tools_have_no_hits(monkeypatch):
    _reset_store()

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            return {"foods": []}

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            return {"results": []}

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())
    monkeypatch.setattr(
        "app.nutrition.meal_log_service._infer_nutrition_with_llm",
        lambda _: (NutritionTotals(energy_kcal=480, protein_g=17, carbs_g=64, fat_g=16), 0.45),
    )

    draft = prepare_meal_log_draft(
        user_id="tg:6",
        chat_id="tg:chat6",
        input_source=InputSource.text,
        meal_description="some unknown dish",
    )
    totals = draft["nutrition_totals"]
    assert totals["energy_kcal"] == 480
    assert totals["protein_g"] == 17
