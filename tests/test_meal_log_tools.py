import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.tools.meal_log_tools import prepare_meal_log


def test_prepare_meal_log_forwards_final_estimate(monkeypatch):
    captured = {}

    def fake_prepare_meal_log_draft(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "draft_id": "draft-1", "nutrition_source": kwargs["nutrition_source"]}

    monkeypatch.setattr("app.tools.meal_log_tools.prepare_meal_log_draft", fake_prepare_meal_log_draft)

    result = prepare_meal_log.func(
        user_id="tg:tool-1",
        chat_id="tg:chat-tool-1",
        meal_description="晚餐吃了意大利面",
        energy_kcal=520,
        protein_g=24,
        carbs_g=58,
        fat_g=18,
        nutrition_source="Spoonacular",
        nutrition_confidence=0.74,
    )

    assert result["ok"] is True
    assert captured["meal_description"] == "晚餐吃了意大利面"
    assert captured["nutrition_totals"].model_dump() == {
        "energy_kcal": 520.0,
        "protein_g": 24.0,
        "carbs_g": 58.0,
        "fat_g": 18.0,
        "fiber_g": None,
        "sugar_g": None,
        "sodium_mg": None,
        "cholesterol_mg": None,
        "water_ml": None,
        "micronutrients": {},
    }
    assert captured["nutrition_source"] == "Spoonacular"
    assert captured["nutrition_confidence"] == 0.74
