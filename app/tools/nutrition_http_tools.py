import os
from typing import Any

import httpx
from langchain_core.tools import tool

from app.tracing import trace_log


SPOONACULAR_BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
TAVILY_SEARCH_BASE_URL = "https://api.tavily.com/search"
OPEN_FOOD_FACTS_SEARCH_BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OPENFOODFACTS_USER_AGENT = os.getenv(
    "OPENFOODFACTS_USER_AGENT",
    "ai-diet-assistant/0.1 (contact: dev@example.com)",
)


def _truncate_text(value: Any, limit: int = 160) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _trace_tool_call(tool_name: str, payload: dict[str, Any]) -> None:
    trace_log("ToolCall", f"{tool_name} invoked", payload)


def _trace_tool_result(tool_name: str, query: str, results: list[dict[str, Any]], count: int) -> None:
    trace_log(
        "ToolResult",
        f"{tool_name} returned",
        {
            "query": query,
            "count": count,
            "preview": results[:3],
        },
    )


@tool(
    description=(
        "Search recipe-style dishes and composed meals from Spoonacular with nutrition summaries. "
        "Best for plated dishes, home-style meals, and named recipes such as spaghetti bolognese or chicken curry."
    )
)
def spoonacular_search_recipe(
    query: str,
    diet: str | None = None,
    intolerances: str | None = None,
    number: int = 5,
) -> dict[str, Any]:
    api_key = os.getenv("SPOONACULAR_API_KEY", "").strip()
    if not api_key:
        return {"error": "missing SPOONACULAR_API_KEY"}

    _trace_tool_call(
        "spoonacular_search_recipe",
        {
            "query": query,
            "diet": diet,
            "intolerances": intolerances,
            "number": max(1, min(number, 20)),
        },
    )

    params = {
        "query": query,
        "diet": diet,
        "intolerances": intolerances,
        "addRecipeNutrition": True,
        "number": max(1, min(number, 20)),
        "apiKey": api_key,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(SPOONACULAR_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return {"error": f"spoonacular request failed: {exc}"}

    results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        nutrients = {
            n.get("name", "").lower(): n
            for n in item.get("nutrition", {}).get("nutrients", [])
            if isinstance(n, dict)
        }

        def nutrient_value(name: str) -> float | None:
            val = nutrients.get(name.lower(), {}).get("amount")
            return val if isinstance(val, (int, float)) else None

        title = item.get("title", "recipe")
        rid = item.get("id")
        slug = str(title).replace(" ", "-")
        url = f"https://spoonacular.com/recipes/{slug}-{rid}" if rid else None

        results.append(
            {
                "id": rid,
                "title": title,
                "image": item.get("image"),
                "calories": nutrient_value("Calories"),
                "protein_g": nutrient_value("Protein"),
                "carbs_g": nutrient_value("Carbohydrates"),
                "fat_g": nutrient_value("Fat"),
                "url": url,
            }
        )

    preview = [
        {
            "title": item.get("title"),
            "calories": item.get("calories"),
            "protein_g": item.get("protein_g"),
            "carbs_g": item.get("carbs_g"),
            "fat_g": item.get("fat_g"),
        }
        for item in results
    ]
    _trace_tool_result("spoonacular_search_recipe", query, preview, len(results))

    return {"results": results, "count": len(results), "source": "spoonacular"}


@tool(
    description=(
        "Search USDA FoodData Central foods and return key nutrient values. "
        "Best for generic ingredients, simple foods, and common staples such as rice, eggs, chicken breast, or yogurt."
    )
)
def usda_search_foods(
    query: str,
    page_size: int = 5,
    page_number: int = 1,
) -> dict[str, Any]:
    api_key = os.getenv("USDA_API_KEY", "").strip()
    if not api_key:
        return {"error": "missing USDA_API_KEY"}

    _trace_tool_call(
        "usda_search_foods",
        {
            "query": query,
            "page_size": max(1, min(page_size, 25)),
            "page_number": max(1, page_number),
        },
    )

    payload = {
        "query": query,
        "pageSize": max(1, min(page_size, 25)),
        "pageNumber": max(1, page_number),
    }
    params = {"api_key": api_key}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(USDA_BASE_URL, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return {"error": f"usda request failed: {exc}"}

    foods = []
    for item in data.get("foods", []):
        nutrients: dict[str, float] = {}
        for n in item.get("foodNutrients", []):
            if not isinstance(n, dict):
                continue
            n_name = n.get("nutrientName")
            n_value = n.get("value")
            if isinstance(n_name, str) and isinstance(n_value, (int, float)):
                nutrients[n_name] = n_value

        foods.append(
            {
                "fdc_id": item.get("fdcId"),
                "description": item.get("description"),
                "data_type": item.get("dataType"),
                "brand_name": item.get("brandName"),
                "serving_size": item.get("servingSize"),
                "serving_size_unit": item.get("servingSizeUnit"),
                "calories_kcal": nutrients.get("Energy"),
                "protein_g": nutrients.get("Protein"),
                "carbs_g": nutrients.get("Carbohydrate, by difference"),
                "fat_g": nutrients.get("Total lipid (fat)"),
            }
        )

    preview = [
        {
            "description": item.get("description"),
            "brand_name": item.get("brand_name"),
            "calories_kcal": item.get("calories_kcal"),
            "protein_g": item.get("protein_g"),
            "carbs_g": item.get("carbs_g"),
            "fat_g": item.get("fat_g"),
        }
        for item in foods
    ]
    _trace_tool_result("usda_search_foods", query, preview, len(foods))

    return {
        "foods": foods,
        "count": len(foods),
        "total_hits": data.get("totalHits"),
        "source": "usda_food_data_central",
    }


@tool(
    description=(
        "Search the web with Tavily for nutrition clues when structured food databases are a poor fit. "
        "Use as a fallback for restaurant meals, niche foods, or cases where USDA, Spoonacular, and Open Food Facts are not sufficient."
    )
)
def tavily_search_nutrition(query: str, max_results: int = 5) -> dict[str, Any]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"error": "missing TAVILY_API_KEY"}

    resolved_query = f"{query} nutrition facts calories protein carbs fat"
    _trace_tool_call(
        "tavily_search_nutrition",
        {
            "query": query,
            "resolved_query": resolved_query,
            "max_results": max(1, min(max_results, 10)),
        },
    )

    payload = {
        "api_key": api_key,
        "query": resolved_query,
        "search_depth": "basic",
        "max_results": max(1, min(max_results, 10)),
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(TAVILY_SEARCH_BASE_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return {"error": f"tavily request failed: {exc}"}

    results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
        )

    preview = [
        {
            "title": item.get("title"),
            "score": item.get("score"),
            "url": item.get("url"),
            "content": _truncate_text(item.get("content")),
        }
        for item in results
    ]
    _trace_tool_result("tavily_search_nutrition", query, preview, len(results))

    return {
        "results": results,
        "count": len(results),
        "source": "tavily",
    }


@tool(
    description=(
        "Search Open Food Facts for packaged or branded food products and nutrition values. "
        "Best for store-bought bars, drinks, snacks, cereal, and other packaged consumer foods."
    )
)
def openfoodfacts_search_products(query: str, page_size: int = 5) -> dict[str, Any]:
    _trace_tool_call(
        "openfoodfacts_search_products",
        {
            "query": query,
            "page_size": max(1, min(page_size, 10)),
        },
    )

    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": max(1, min(page_size, 10)),
        "fields": ",".join(
            [
                "code",
                "product_name",
                "brands",
                "serving_size",
                "nutriments",
            ]
        ),
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                OPEN_FOOD_FACTS_SEARCH_BASE_URL,
                params=params,
                headers={"User-Agent": OPENFOODFACTS_USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        return {"error": f"openfoodfacts request failed: {exc}"}

    products: list[dict[str, Any]] = []
    for item in data.get("products", []):
        if not isinstance(item, dict):
            continue
        nutriments = item.get("nutriments") or {}
        if not isinstance(nutriments, dict):
            nutriments = {}
        products.append(
            {
                "code": item.get("code"),
                "product_name": item.get("product_name"),
                "brands": item.get("brands"),
                "serving_size": item.get("serving_size"),
                "calories_kcal": nutriments.get("energy-kcal_serving")
                or nutriments.get("energy-kcal_100g"),
                "protein_g": nutriments.get("proteins_serving") or nutriments.get("proteins_100g"),
                "carbs_g": nutriments.get("carbohydrates_serving") or nutriments.get("carbohydrates_100g"),
                "fat_g": nutriments.get("fat_serving") or nutriments.get("fat_100g"),
            }
        )

    preview = [
        {
            "product_name": item.get("product_name"),
            "brands": item.get("brands"),
            "serving_size": item.get("serving_size"),
            "calories_kcal": item.get("calories_kcal"),
            "protein_g": item.get("protein_g"),
            "carbs_g": item.get("carbs_g"),
            "fat_g": item.get("fat_g"),
        }
        for item in products
    ]
    _trace_tool_result("openfoodfacts_search_products", query, preview, len(products))

    return {
        "products": products,
        "count": len(products),
        "source": "openfoodfacts",
    }
