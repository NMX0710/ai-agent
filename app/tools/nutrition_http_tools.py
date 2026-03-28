import os
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

from app.observability import trace_log


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


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _looks_like_composed_dish(text: Any) -> bool:
    normalized = _normalize_text(text)
    keywords = (
        " curry",
        "sandwich",
        "burger",
        "roll",
        "sushi",
        "pasta",
        "spaghetti",
        "fried rice",
        "rice",
        "bowl",
        "plate",
        "wrap",
    )
    return any(keyword in normalized for keyword in keywords)


def _macro_consistency_warning(
    *,
    calories_kcal: float | None,
    protein_g: float | None,
    carbs_g: float | None,
    fat_g: float | None,
) -> str | None:
    values = (calories_kcal, protein_g, carbs_g, fat_g)
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    derived_kcal = (protein_g * 4) + (carbs_g * 4) + (fat_g * 9)
    if derived_kcal <= 0:
        return None
    delta = abs(calories_kcal - derived_kcal)
    tolerance = max(120.0, max(calories_kcal, derived_kcal) * 0.35)
    if delta <= tolerance:
        return None
    return (
        "Calories and macros look internally inconsistent for one serving. "
        "Use caution before treating this candidate as a final answer."
    )


def _extract_domain(url: Any) -> str | None:
    parsed = urlparse(str(url or "").strip())
    return parsed.netloc.lower() or None


def _usda_basis_warning(item: dict[str, Any]) -> str | None:
    description = item.get("description")
    serving_size = item.get("servingSize")
    if serving_size is not None:
        return None
    if not _looks_like_composed_dish(description):
        return None
    return (
        "Serving basis is unclear for this composed dish entry because USDA did not provide a serving size. "
        "Do not assume this is a full plated serving; it may reflect a small portion or low-granularity entry."
    )


@tool(
    description=(
        "Search recipe-style dishes and composed meals from Spoonacular with nutrition summaries. "
        "Best for plated dishes, home-style meals, and named recipes such as spaghetti bolognese or chicken curry. "
        "For tool calls, the query must be an English dish or recipe name."
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
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        return {
            "error": f"spoonacular request failed: {exc}",
            "source": "spoonacular",
            "unavailable": True,
            "error_type": "quota_or_access" if status_code in {401, 402, 403} else "http_error",
            "fallback_recommendation": "skip_recipe_lookup_and_use_decomposition_or_other_source",
        }
    except Exception as exc:
        return {
            "error": f"spoonacular request failed: {exc}",
            "source": "spoonacular",
            "unavailable": True,
            "error_type": "request_failed",
            "fallback_recommendation": "skip_recipe_lookup_and_use_decomposition_or_other_source",
        }

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
        calories = nutrient_value("Calories")
        protein_g = nutrient_value("Protein")
        carbs_g = nutrient_value("Carbohydrates")
        fat_g = nutrient_value("Fat")
        consistency_warning = _macro_consistency_warning(
            calories_kcal=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
        )
        candidate_warnings = [warning for warning in (consistency_warning,) if warning]

        results.append(
            {
                "id": rid,
                "title": title,
                "image": item.get("image"),
                "calories": calories,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fat_g": fat_g,
                "url": url,
                "serving_basis": "recipe_serving",
                "is_composed_dish": True,
                "candidate_warnings": candidate_warnings,
                "internal_consistency_warning": consistency_warning,
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

    return {
        "results": results,
        "count": len(results),
        "source": "spoonacular",
    }


@tool(
    description=(
        "Search USDA FoodData Central foods and return key nutrient values. "
        "Best for generic ingredients, simple foods, and common staples such as rice, eggs, chicken breast, or yogurt. "
        "Do not use this as the primary lookup path for named restaurant menu items or exact branded prepared meals such as Subway sandwiches or Big Mac. "
        "For tool calls, the query must be an English common food or ingredient name."
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

        basis_warning = _usda_basis_warning(item)
        description = item.get("description")
        brand_name = item.get("brandName")
        serving_size = item.get("servingSize")
        calories_kcal = nutrients.get("Energy")
        protein_g = nutrients.get("Protein")
        carbs_g = nutrients.get("Carbohydrate, by difference")
        fat_g = nutrients.get("Total lipid (fat)")
        is_composed_dish = _looks_like_composed_dish(description)
        serving_basis_unclear = basis_warning is not None
        candidate_warnings = [
            warning
            for warning in (
                basis_warning,
                _macro_consistency_warning(
                    calories_kcal=calories_kcal,
                    protein_g=protein_g,
                    carbs_g=carbs_g,
                    fat_g=fat_g,
                ),
            )
            if warning
        ]
        foods.append(
            {
                "fdc_id": item.get("fdcId"),
                "description": description,
                "data_type": item.get("dataType"),
                "brand_name": brand_name,
                "serving_size": serving_size,
                "serving_size_unit": item.get("servingSizeUnit"),
                "calories_kcal": calories_kcal,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fat_g": fat_g,
                "basis_warning": basis_warning,
                "is_composed_dish": is_composed_dish,
                "serving_basis_unclear": serving_basis_unclear,
                "candidate_warnings": candidate_warnings,
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
            "basis_warning": item.get("basis_warning"),
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
        "Use as a fallback for restaurant meals, chain menu items, niche foods, or cases where USDA, Spoonacular, and Open Food Facts are not sufficient. "
        "For named restaurant items, prefer official restaurant pages or snippets that clearly match the exact menu item."
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
        domain = _extract_domain(item.get("url"))
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "source_domain": domain,
                "content": item.get("content"),
                "score": item.get("score"),
                "usage_note": (
                    "Web snippets are fallback clues, not canonical structured nutrition totals. "
                    "Prefer an official brand or restaurant page when one clearly matches the named item."
                ),
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
        "Best for exact packaged or branded products such as store-bought bars, drinks, snacks, cereal, and frozen prepared meals. "
        "For an exact branded packaged item, use this before generic recipe or ingredient databases."
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
        calories_serving = nutriments.get("energy-kcal_serving")
        calories_100g = nutriments.get("energy-kcal_100g")
        protein_serving = nutriments.get("proteins_serving")
        protein_100g = nutriments.get("proteins_100g")
        carbs_serving = nutriments.get("carbohydrates_serving")
        carbs_100g = nutriments.get("carbohydrates_100g")
        fat_serving = nutriments.get("fat_serving")
        fat_100g = nutriments.get("fat_100g")
        nutrition_basis = "serving" if calories_serving is not None else "per_100g_fallback"
        calories_kcal = calories_serving or calories_100g
        protein_g = protein_serving or protein_100g
        carbs_g = carbs_serving or carbs_100g
        fat_g = fat_serving or fat_100g
        candidate_warnings = [
            warning
            for warning in (
                (
                    "Nutrition values fell back to per-100g data because no serving-level values were present."
                    if nutrition_basis != "serving"
                    else None
                ),
                _macro_consistency_warning(
                    calories_kcal=calories_kcal,
                    protein_g=protein_g,
                    carbs_g=carbs_g,
                    fat_g=fat_g,
                ),
            )
            if warning
        ]
        products.append(
            {
                "code": item.get("code"),
                "product_name": item.get("product_name"),
                "brands": item.get("brands"),
                "serving_size": item.get("serving_size"),
                "calories_kcal": calories_kcal,
                "protein_g": protein_g,
                "carbs_g": carbs_g,
                "fat_g": fat_g,
                "nutrition_basis": nutrition_basis,
                "candidate_warnings": candidate_warnings,
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
            "nutrition_basis": item.get("nutrition_basis"),
        }
        for item in products
    ]
    _trace_tool_result("openfoodfacts_search_products", query, preview, len(products))

    return {
        "products": products,
        "count": len(products),
        "source": "openfoodfacts",
    }
