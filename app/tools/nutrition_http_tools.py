import os
import re
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

_STOPWORDS = {
    "a",
    "an",
    "and",
    "bar",
    "brand",
    "burger",
    "calories",
    "carbs",
    "chicken",
    "cookie",
    "dough",
    "fat",
    "facts",
    "food",
    "for",
    "fresh",
    "frozen",
    "grams",
    "inch",
    "large",
    "macros",
    "meal",
    "menu",
    "nutrition",
    "of",
    "one",
    "orange",
    "pack",
    "per",
    "plate",
    "pork",
    "prepared",
    "product",
    "protein",
    "restaurant",
    "roll",
    "sandwich",
    "serving",
    "size",
    "small",
    "sushi",
    "the",
    "turkey",
    "with",
}

_NAME_STOPWORDS = {"a", "an", "and", "of", "the", "with"}


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


def _tokenize_meaningful_words(value: Any) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9']+", _normalize_text(value))
        if len(token) > 1 and token not in _STOPWORDS
    }
    return tokens


def _tokenize_name_terms(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", _normalize_text(value))
        if len(token) > 1 and token not in _NAME_STOPWORDS
    }


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


def _food_form_hint(text: Any) -> str | None:
    normalized = _normalize_text(text)
    if "big mac" in normalized or "burger" in normalized:
        return "burger"
    if "sandwich" in normalized or "sub" in normalized:
        return "sandwich"
    if "california roll" in normalized or "sushi roll" in normalized or "sushi" in normalized:
        return "sushi_roll"
    if "pasta" in normalized or "spaghetti" in normalized:
        return "pasta_plate"
    if "fried rice" in normalized:
        return "fried_rice"
    if "curry rice" in normalized or "curry with rice" in normalized or ("curry" in normalized and "rice" in normalized):
        return "curry_rice"
    if "protein bar" in normalized or normalized.endswith(" bar") or "quest" in normalized:
        return "protein_bar"
    return None


def _query_context(query: str) -> dict[str, Any]:
    normalized = _normalize_text(query)
    food_form = _food_form_hint(normalized)
    likely_full_serving_request = food_form in {
        "burger",
        "sandwich",
        "sushi_roll",
        "pasta_plate",
        "fried_rice",
        "curry_rice",
        "protein_bar",
    }
    likely_restaurant_or_menu_item = any(
        phrase in normalized
        for phrase in ("big mac", "subway", "mcdonald", "menu", "restaurant")
    )
    likely_branded_packaged_food = any(
        phrase in normalized
        for phrase in ("quest", "trader joe", "protein bar", "frozen", "mandarin orange chicken")
    )
    mandatory_components: list[str] = []
    if food_form == "sandwich":
        mandatory_components.append("bread")
    if food_form == "burger":
        mandatory_components.extend(["bun", "patty"])
    if food_form == "sushi_roll":
        mandatory_components.extend(["rice", "nori", "filling"])
    return {
        "food_form": food_form,
        "likely_full_serving_request": likely_full_serving_request,
        "likely_restaurant_or_menu_item": likely_restaurant_or_menu_item,
        "likely_branded_packaged_food": likely_branded_packaged_food,
        "mandatory_components": mandatory_components,
    }


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
        "Do not treat this candidate as a reliable final answer without a better source."
    )


def _shared_brand_tokens(query: str, *values: Any) -> list[str]:
    query_tokens = _tokenize_meaningful_words(query)
    candidate_tokens: set[str] = set()
    for value in values:
        candidate_tokens.update(_tokenize_meaningful_words(value))
    return sorted(query_tokens & candidate_tokens)


def _name_match_score(query: str, *values: Any) -> float:
    query_tokens = _tokenize_name_terms(query)
    if not query_tokens:
        return 0.0
    candidate_tokens: set[str] = set()
    for value in values:
        candidate_tokens.update(_tokenize_name_terms(value))
    if not candidate_tokens:
        return 0.0
    return round(len(query_tokens & candidate_tokens) / len(query_tokens), 3)


def _style_modifier_warning(query: str, candidate_text: Any) -> str | None:
    normalized_query = _normalize_text(query)
    normalized_candidate = _normalize_text(candidate_text)
    style_terms = ("korean", "japanese", "thai", "cauliflower", "pork", "creamy", "mulligatawny")
    unexpected = [term for term in style_terms if term in normalized_candidate and term not in normalized_query]
    if not unexpected:
        return None
    return (
        "Candidate adds style or ingredient modifiers not requested by the user: "
        + ", ".join(unexpected)
        + ". Prefer a more canonical match if available."
    )


def _dish_form_macro_warning(
    food_form: str | None,
    *,
    calories_kcal: float | None,
    protein_g: float | None,
    carbs_g: float | None,
) -> str | None:
    if food_form is None:
        return None
    thresholds = {
        "curry_rice": {"calories": 350, "protein": 8, "carbs": 25},
        "fried_rice": {"calories": 250, "protein": 10, "carbs": 25},
        "sandwich": {"calories": 220, "protein": 10, "carbs": 20},
        "burger": {"calories": 300, "protein": 12, "carbs": 20},
        "sushi_roll": {"calories": 180, "protein": 4, "carbs": 20},
        "pasta_plate": {"calories": 250, "protein": 8, "carbs": 35},
        "protein_bar": {"calories": 120, "protein": 10, "carbs": 10},
    }
    expected = thresholds.get(food_form)
    if not expected:
        return None
    checks = (
        ("calories", calories_kcal, expected["calories"]),
        ("protein", protein_g, expected["protein"]),
        ("carbs", carbs_g, expected["carbs"]),
    )
    failed = [label for label, value, minimum in checks if isinstance(value, (int, float)) and value < minimum]
    if not failed:
        return None
    return (
        f"This candidate looks too small or unbalanced for a typical full-serving {food_form.replace('_', ' ')}. "
        f"Low fields: {', '.join(failed)}."
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

    query_context = _query_context(query)
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
        title_food_form = _food_form_hint(title)
        dish_form_match = (
            True
            if not query_context["food_form"]
            else title_food_form == query_context["food_form"]
        )
        name_match_score = _name_match_score(query, title)
        macro_warning = _dish_form_macro_warning(
            query_context["food_form"],
            calories_kcal=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
        )
        style_warning = _style_modifier_warning(query, title)
        candidate_warnings = [
            warning
            for warning in (
                (
                    "Recipe title drifts into a different dish form than the user asked for."
                    if not dish_form_match
                    else None
                ),
                (
                    "Recipe title has weak name overlap with the requested dish."
                    if query_context["food_form"] and name_match_score < 0.35
                    else None
                ),
                style_warning,
                macro_warning,
                consistency_warning,
            )
            if warning
        ]
        not_recommended_for_full_serving_estimate = bool(
            query_context["likely_full_serving_request"]
            and (
                not dish_form_match
                or macro_warning is not None
                or (query_context["food_form"] and name_match_score < 0.35)
            )
        )

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
                "title_food_form": title_food_form,
                "dish_form_match": dish_form_match,
                "name_match_score": name_match_score,
                "not_recommended_for_full_serving_estimate": not_recommended_for_full_serving_estimate,
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
        "query_context": query_context,
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

    query_context = _query_context(query)
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
        brand_match_tokens = _shared_brand_tokens(query, description, brand_name)
        brand_match_confident = bool(brand_match_tokens)
        serving_basis_unclear = basis_warning is not None
        likely_per_100g_or_small_portion = bool(
            is_composed_dish and serving_basis_unclear and isinstance(calories_kcal, (int, float)) and calories_kcal <= 150
        )
        low_full_serving_threshold = {
            "burger": 250,
            "sandwich": 220,
            "sushi_roll": 150,
            "pasta_plate": 220,
            "fried_rice": 220,
            "curry_rice": 220,
        }.get(query_context["food_form"], 180)
        not_recommended_for_full_serving_estimate = bool(
            query_context["likely_full_serving_request"]
            and is_composed_dish
            and (
                serving_basis_unclear
                or likely_per_100g_or_small_portion
                or (isinstance(calories_kcal, (int, float)) and calories_kcal <= low_full_serving_threshold)
            )
        )
        restaurant_query_mismatch = bool(
            query_context["likely_restaurant_or_menu_item"] and not brand_match_confident
        )
        not_recommended_for_full_serving_estimate = bool(
            not_recommended_for_full_serving_estimate or restaurant_query_mismatch
        )
        exposed_calories_kcal = None if restaurant_query_mismatch else calories_kcal
        exposed_protein_g = None if restaurant_query_mismatch else protein_g
        exposed_carbs_g = None if restaurant_query_mismatch else carbs_g
        exposed_fat_g = None if restaurant_query_mismatch else fat_g
        candidate_warnings = [
            warning
            for warning in (
                basis_warning,
                (
                    "This candidate looks too small or too low-granularity for a full-serving answer."
                    if likely_per_100g_or_small_portion
                    else None
                ),
                (
                    "Do not use this USDA row as the final answer for a full-serving estimate."
                    if not_recommended_for_full_serving_estimate
                    else None
                ),
                (
                    "This USDA row does not look restaurant-aware or brand-matched for the named menu item."
                    if restaurant_query_mismatch
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
        foods.append(
            {
                "fdc_id": item.get("fdcId"),
                "description": description,
                "data_type": item.get("dataType"),
                "brand_name": brand_name,
                "serving_size": serving_size,
                "serving_size_unit": item.get("servingSizeUnit"),
                "calories_kcal": exposed_calories_kcal,
                "protein_g": exposed_protein_g,
                "carbs_g": exposed_carbs_g,
                "fat_g": exposed_fat_g,
                "generic_reference_nutrition": (
                    {
                        "calories_kcal": calories_kcal,
                        "protein_g": protein_g,
                        "carbs_g": carbs_g,
                        "fat_g": fat_g,
                    }
                    if restaurant_query_mismatch
                    else None
                ),
                "basis_warning": basis_warning,
                "is_composed_dish": is_composed_dish,
                "brand_match_tokens": brand_match_tokens,
                "brand_match_confident": brand_match_confident,
                "serving_basis_unclear": serving_basis_unclear,
                "likely_per_100g_or_small_portion": likely_per_100g_or_small_portion,
                "restaurant_query_mismatch": restaurant_query_mismatch,
                "not_recommended_for_full_serving_estimate": not_recommended_for_full_serving_estimate,
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
            "brand_match_confident": item.get("brand_match_confident"),
            "not_recommended_for_full_serving_estimate": item.get("not_recommended_for_full_serving_estimate"),
        }
        for item in foods
    ]
    _trace_tool_result("usda_search_foods", query, preview, len(foods))

    return {
        "foods": foods,
        "count": len(foods),
        "total_hits": data.get("totalHits"),
        "source": "usda_food_data_central",
        "query_context": query_context,
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

    query_context = _query_context(query)
    results: list[dict[str, Any]] = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        domain = _extract_domain(item.get("url"))
        shared_brand_tokens = _shared_brand_tokens(query, item.get("title"), domain)
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "source_domain": domain,
                "content": item.get("content"),
                "score": item.get("score"),
                "brand_match_tokens": shared_brand_tokens,
                "looks_like_official_brand_or_restaurant_page": bool(
                    domain and any(token in domain for token in shared_brand_tokens)
                ),
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
        "query_context": query_context,
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

    query_context = _query_context(query)
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
        shared_brand_tokens = _shared_brand_tokens(query, item.get("product_name"), item.get("brands"))
        brand_match_confident = bool(shared_brand_tokens)
        name_match_score = _name_match_score(query, item.get("product_name"), item.get("brands"))
        exact_name_match = name_match_score >= 0.6
        not_recommended_for_full_serving_estimate = bool(
            query_context["likely_full_serving_request"]
            and nutrition_basis != "serving"
            and not item.get("serving_size")
        )
        candidate_warnings = [
            warning
            for warning in (
                (
                    "Nutrition values fell back to per-100g data because no serving-level values were present."
                    if nutrition_basis != "serving"
                    else None
                ),
                (
                    "This product row is not ideal for a one-item final answer because serving size is missing."
                    if not_recommended_for_full_serving_estimate
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
                "brand_match_tokens": shared_brand_tokens,
                "brand_match_confident": brand_match_confident,
                "name_match_score": name_match_score,
                "exact_name_match": exact_name_match,
                "not_recommended_for_full_serving_estimate": not_recommended_for_full_serving_estimate,
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
            "brand_match_confident": item.get("brand_match_confident"),
            "exact_name_match": item.get("exact_name_match"),
        }
        for item in products
    ]
    _trace_tool_result("openfoodfacts_search_products", query, preview, len(products))

    return {
        "products": products,
        "count": len(products),
        "source": "openfoodfacts",
        "query_context": query_context,
    }
