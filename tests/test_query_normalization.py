import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.nutrition.meallog import InputSource, NutritionTotals
from app.nutrition.meal_log_service import prepare_meal_log_draft
from app.nutrition.query_normalization import normalize_meal_query


def test_normalize_meal_query_strips_chinese_wrapper_and_builds_english_query():
    normalized = normalize_meal_query(user_text="我晚上吃了意大利面，可以帮我记录吗？")

    assert normalized.meal_description == "我晚上吃了意大利面，可以帮我记录吗？"
    assert normalized.food_query == "意大利面"
    assert normalized.food_query_en == "spaghetti"
    assert normalized.detected_language == "zh"


def test_normalize_meal_query_keeps_explicit_food_query_over_wrapper_text():
    normalized = normalize_meal_query(
        user_text="帮我记录一下，我晚上吃了意大利面",
        meal_description="我晚上吃了意大利面",
        food_query="番茄肉酱意大利面",
        food_query_en="spaghetti bolognese",
    )

    assert normalized.food_query == "番茄肉酱意大利面"
    assert normalized.food_query_en == "spaghetti bolognese"


def test_normalize_meal_query_strips_english_wrapper_text():
    normalized = normalize_meal_query(user_text="I had chicken rice tonight, please log it")

    assert normalized.food_query == "chicken rice"
    assert normalized.food_query_en == "chicken rice"
    assert normalized.detected_language == "en"


def test_prepare_meal_log_draft_prefers_english_lookup_query_for_chinese_input(monkeypatch):
    captured: dict[str, str] = {}

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            captured["query"] = payload["query"]
            return {
                "foods": [
                    {"calories_kcal": 510, "protein_g": 20, "carbs_g": 70, "fat_g": 12},
                ]
            }

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            raise AssertionError("Spoonacular should not be called when USDA already returns usable macros.")

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())

    draft = prepare_meal_log_draft(
        user_id="tg:query-1",
        chat_id="tg:chat-query-1",
        input_source=InputSource.text,
        meal_description="我晚上吃了意大利面，可以帮我记录吗？",
    )

    assert captured["query"] == "spaghetti"
    assert draft["food_query"] == "意大利面"
    assert draft["food_query_en"] == "spaghetti"
    assert draft["nutrition_totals"] == NutritionTotals(
        energy_kcal=510,
        protein_g=20,
        carbs_g=70,
        fat_g=12,
    ).model_dump()


def test_prepare_meal_log_draft_falls_back_to_original_food_query_when_translation_is_unknown(monkeypatch):
    captured: list[str] = []

    class _FakeUsdaTool:
        @staticmethod
        def invoke(payload):
            captured.append(payload["query"])
            return {"foods": []}

    class _FakeSpoonacularTool:
        @staticmethod
        def invoke(payload):
            captured.append(payload["query"])
            return {
                "results": [
                    {"calories": 420, "protein_g": 16, "carbs_g": 58, "fat_g": 14},
                ]
            }

    monkeypatch.setattr("app.nutrition.meal_log_service.usda_search_foods", _FakeUsdaTool())
    monkeypatch.setattr("app.nutrition.meal_log_service.spoonacular_search_recipe", _FakeSpoonacularTool())

    draft = prepare_meal_log_draft(
        user_id="tg:query-2",
        chat_id="tg:chat-query-2",
        input_source=InputSource.text,
        meal_description="记录一下藜麦鸡肉能量碗",
    )

    assert captured[0] == "藜麦鸡肉能量碗"
    assert captured == ["藜麦鸡肉能量碗", "藜麦鸡肉能量碗"]
    assert draft["food_query"] == "藜麦鸡肉能量碗"
    assert draft["food_query_en"] == "藜麦鸡肉能量碗"


def test_normalize_meal_query_preserves_agent_supplied_queries():
    normalized = normalize_meal_query(
        user_text="我晚上吃了意大利面，可以帮我记录吗",
        meal_description="晚餐吃了意大利面",
        food_query="意大利面",
        food_query_en="Spaghetti",
    )

    assert normalized.meal_description == "晚餐吃了意大利面"
    assert normalized.food_query == "意大利面"
    assert normalized.food_query_en == "spaghetti"
