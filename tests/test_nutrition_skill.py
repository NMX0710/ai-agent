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

    assert "prefer recipe-style lookup with Spoonacular first" in content
    assert "Only fall back to ingredient decomposition" in content
    assert "matches the user's portion level and meal context" in content
    assert "Do not treat a likely per-100g value" in content
    assert "ask one concise, high-value clarification question" in content
    assert "must use nutrition lookup tools before returning a nutrition estimate" in content
    assert "do not ask for grams by default; use a common single-serving decomposition" in content
    assert "use the restaurant-aware Tavily path before Open Food Facts or generic USDA rows" in content
    assert "choose one representative estimate instead of returning a range" in content
    assert "Return either one concise clarification question or one final nutrition estimate" in content


def test_nutrition_lookup_skill_metadata_and_body_cover_reliability_checks():
    skill_path = Path(project_root) / "app/skills/nutrition/nutrition-lookup/SKILL.md"
    content = skill_path.read_text()

    assert "serving basis and portion level" in content
    assert "Do not answer from general nutrition knowledge alone" in content
    assert "prefer recipe-style lookup for composed dishes before decomposition" in content
    assert "use compact recipe-name queries for Spoonacular" in content
    assert "Estimate Reliability Check" in content
    assert "Tool Priority For Dishes" in content
    assert "Recipe Candidate Matching" in content
    assert "Use a short recipe-style English dish name for Spoonacular" in content
    assert "`chicken curry rice`, `chicken curry`, or `japanese chicken curry`" in content
    assert "Finding a Spoonacular result is not enough by itself." in content
    assert "the title drifts into a different dish type such as `soup`" in content
    assert "extremely low protein for a chicken-and-rice meal" in content
    assert "If the best Spoonacular candidates are only weak matches, do not force one of them." in content
    assert "Creamy Curry Chicken With Yellow Rice" in content
    assert "do not use it as the final answer" in content
    assert "Whole Meal vs Database Entry" in content
    assert "`一份`, `一个`, `一卷`, `one sandwich`, `one burger`, `one roll`" in content
    assert "Restaurant And Brand Policy" in content
    assert "Prefer a brand-aware or restaurant-aware lookup path" in content
    assert "Tool Metadata Interpretation" in content
    assert "`not_recommended_for_full_serving_estimate`" in content
    assert "`nutrition_basis`" in content
    assert "`brand_match_confident`" in content
    assert "`exact_name_match`" in content
    assert "`dish_form_match`" in content
    assert "`name_match_score`" in content
    assert "`restaurant_query_mismatch`" in content
    assert "Dish Decomposition Fallback" in content
    assert "Ingredient Candidate Matching" in content
    assert "Mandatory Component Sanity Checks" in content
    assert "Sandwich: include bread by default." in content
    assert "Sushi roll: treat `one roll`, `一卷`, or `一份加州卷` as a full roll" in content
    assert "Do not default to ingredient decomposition for every dish-level request." in content
    assert "Plan the smallest useful ingredient set before calling tools" in content
    assert "do not treat `usda_search_foods` as a second recipe database for the whole dish name" in content
    assert "Use `tavily_search_nutrition` as a higher-value fallback" in content
    assert "prefer the canonical match rather than returning a range by default" in content
    assert "prefer official brand or restaurant pages when Tavily surfaces them" in content
    assert "chicken breast cooked` should not become breaded tenders or nuggets" in content
    assert "chicken curry rice: rice, chicken, carrots, curry sauce or roux, cooking oil" in content
    assert "California roll: sushi rice, imitation crab or crab mix, avocado, cucumber, nori" in content
    assert "USDA returns a generic row such as `Chicken curry with rice` at 116 kcal" in content
    assert "values roughly in the 400 to 600 kcal range are often more credible" in content
    assert "prefer a common single-serving estimate over a follow-up question" in content
    assert "default to a common single serving instead of asking for grams" in content
    assert "Do not block on a grams question unless the user explicitly wants high precision" in content
    assert "If Spoonacular only returns weak curry-like recipes with implausible macros for chicken curry rice, reject them and move to decomposition" in content
    assert "Do not return a range when one common serving estimate is already good enough" in content
    assert "choose one representative serving-level estimate instead of surfacing the whole band" in content
    assert "Sushi Roll Example" in content
    assert "Return about 94 kcal as the answer for one full California roll" in content
    assert "a result in the several-hundred-kcal range is usually more credible than ~100 kcal" in content
    assert "estimate from sushi rice, crab mix, avocado, cucumber, and nori for one standard roll" in content
    assert "Fast Food Example" in content
    assert "Recognize `Big Mac` as a specific chain menu item" in content
    assert "Restaurant Sandwich Example" in content
    assert "Subway 6-inch turkey sandwich 的热量是多少" in content
    assert "Branded Product Example" in content
    assert "Prefer a branded packaged-food result with a confident Quest match" in content
    assert "choose the strongest exact-name serving-level row instead of returning a range" in content
    assert 'User: "咖喱饭的热量是多少？我放了胡萝卜和鸡肉"' in content
