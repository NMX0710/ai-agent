import logging
from typing import List

from app.tools.meal_log_tools import commit_meal_log, prepare_meal_log
from app.tools.nutrition_http_tools import (
    openfoodfacts_search_products,
    spoonacular_search_recipe,
    tavily_search_nutrition,
    usda_search_foods,
)


def load_all_tools() -> List:
    tools = [
        spoonacular_search_recipe,
        usda_search_foods,
        tavily_search_nutrition,
        openfoodfacts_search_products,
        prepare_meal_log,
        commit_meal_log,
    ]
    logging.info("[ToolRegistry] Total tools ready: %d", len(tools))
    for t in tools:
        logging.info("   ├── %s", getattr(t, "name", type(t).__name__))
    return tools
