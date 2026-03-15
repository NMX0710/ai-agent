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
        lambda _: (NutritionTotals(energy_kcal=500, protein_g=30, carbs_g=40, fat_g=20), 0.7, "USDA"),
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
        lambda _: (NutritionTotals(energy_kcal=300, protein_g=10, carbs_g=20, fat_g=15), 0.6, "USDA"),
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
    claimed = list_pending_apple_health_writes(user_id="tg:2")
    assert len(claimed) == 1

    failed = report_apple_health_write_result(
        draft_id=draft_id,
        user_id="tg:2",
        success=False,
        claim_token=claimed[0]["claim_token"],
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


def test_prepare_prefers_clean_agent_supplied_queries(monkeypatch):
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
        meal_description="晚餐吃了意大利面",
        food_query="意大利面",
        food_query_en="spaghetti",
    )
    assert captured["query"] == "spaghetti"


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


def test_prepare_uses_tavily_after_usda_and_spoonacular_miss(monkeypatch):
    _reset_store()

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            return {"foods": []}

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            return {"results": []}

    class _FakeTavilyTool:
        @staticmethod
        def invoke(payload):
            return {
                "results": [
                    {
                        "title": "Brand bar nutrition facts",
                        "content": "Nutrition facts per bar: 210 calories, protein 20 g, carbs 18 g, fat 7 g.",
                    }
                ]
            }

    class _FakeOpenFoodFactsTool:
        @staticmethod
        def invoke(payload):
            raise AssertionError("Open Food Facts should not be called when Tavily already returns usable macros.")

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.tavily_search_nutrition", _FakeTavilyTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.openfoodfacts_search_products", _FakeOpenFoodFactsTool())
    monkeypatch.setattr(
        "app.nutrition.meal_log_service._infer_nutrition_with_llm",
        lambda _: (_ for _ in ()).throw(AssertionError("LLM fallback should not run when Tavily succeeds")),
    )

    draft = prepare_meal_log_draft(
        user_id="tg:7",
        chat_id="tg:chat7",
        input_source=InputSource.text,
        meal_description="brand protein bar",
    )

    assert draft["nutrition_source"] == "Tavily"
    totals = draft["nutrition_totals"]
    assert totals["energy_kcal"] == 210
    assert totals["protein_g"] == 20


def test_prepare_uses_openfoodfacts_after_tavily_miss(monkeypatch):
    _reset_store()

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            return {"foods": []}

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            return {"results": []}

    class _FakeTavilyTool:
        @staticmethod
        def invoke(payload):
            return {"results": []}

    class _FakeOpenFoodFactsTool:
        @staticmethod
        def invoke(payload):
            return {
                "products": [
                    {
                        "product_name": "Quest Protein Bar",
                        "calories_kcal": 190,
                        "protein_g": 20,
                        "carbs_g": 21,
                        "fat_g": 8,
                    }
                ]
            }

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.tavily_search_nutrition", _FakeTavilyTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.openfoodfacts_search_products", _FakeOpenFoodFactsTool())
    monkeypatch.setattr(
        "app.nutrition.meal_log_service._infer_nutrition_with_llm",
        lambda _: (_ for _ in ()).throw(AssertionError("LLM fallback should not run when Open Food Facts succeeds")),
    )

    draft = prepare_meal_log_draft(
        user_id="tg:8",
        chat_id="tg:chat8",
        input_source=InputSource.text,
        meal_description="quest protein bar",
    )

    assert draft["nutrition_source"] == "OpenFoodFacts"
    totals = draft["nutrition_totals"]
    assert totals["energy_kcal"] == 190
    assert totals["protein_g"] == 20
