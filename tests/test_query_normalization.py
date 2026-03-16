import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

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
