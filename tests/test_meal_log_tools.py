import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.tools.meal_log_tools import prepare_meal_log


def test_prepare_meal_log_forwards_structured_queries(monkeypatch):
    captured = {}

    def fake_prepare_meal_log_draft(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "draft_id": "draft-1", "food_query": kwargs["food_query"], "food_query_en": kwargs["food_query_en"]}

    monkeypatch.setattr("app.tools.meal_log_tools.prepare_meal_log_draft", fake_prepare_meal_log_draft)

    result = prepare_meal_log.func(
        user_id="tg:tool-1",
        chat_id="tg:chat-tool-1",
        meal_description="晚餐吃了意大利面",
        food_query="意大利面",
        food_query_en="spaghetti",
    )

    assert result["ok"] is True
    assert captured["meal_description"] == "晚餐吃了意大利面"
    assert captured["food_query"] == "意大利面"
    assert captured["food_query_en"] == "spaghetti"
