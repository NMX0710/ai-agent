import logging
from typing import List

from app.tools.nutrition_http_tools import spoonacular_search_recipe, usda_search_foods


def load_all_tools() -> List:
    tools = [spoonacular_search_recipe, usda_search_foods]
    logging.info("[ToolRegistry] Total tools ready: %d", len(tools))
    for t in tools:
        logging.info("   ├── %s", getattr(t, "name", type(t).__name__))
    return tools
