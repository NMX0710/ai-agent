import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

def test_nutrition_specialist_prompt_includes_reliability_and_decomposition_rules():
    recipe_app_path = Path(project_root) / "app/recipe_app.py"
    content = recipe_app_path.read_text()

    assert "matches the user's portion level and meal context" in content
    assert "Do not treat a likely per-100g value" in content
    assert "decompose the dish into common ingredients" in content
    assert "If an estimate is implausibly low or high" in content


def test_nutrition_lookup_skill_metadata_and_body_cover_reliability_checks():
    skill_path = Path(project_root) / "app/skills/nutrition/nutrition-lookup/SKILL.md"
    content = skill_path.read_text()

    assert "serving basis and portion level" in content
    assert "Estimate Reliability Check" in content
    assert "Whole Meal vs Database Entry" in content
    assert "Dish Decomposition Fallback" in content
    assert "chicken curry rice: rice, chicken, carrots, curry sauce or roux, cooking oil" in content
    assert 'User: "咖喱饭的热量是多少？我放了胡萝卜和鸡肉"' in content
