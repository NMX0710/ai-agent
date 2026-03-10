import os
from typing import Any

import httpx
from langchain_core.tools import tool


SPOONACULAR_BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


@tool(description="Search recipes from Spoonacular with optional nutrition summary.")
def spoonacular_search_recipe(
    query: str,
    diet: str | None = None,
    intolerances: str | None = None,
    number: int = 5,
) -> dict[str, Any]:
    api_key = os.getenv("SPOONACULAR_API_KEY", "").strip()
    if not api_key:
        return {"error": "missing SPOONACULAR_API_KEY"}

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

    return {"results": results, "count": len(results), "source": "spoonacular"}


@tool(description="Search USDA FoodData Central foods and return key nutrient values.")
def usda_search_foods(
    query: str,
    page_size: int = 5,
    page_number: int = 1,
) -> dict[str, Any]:
    api_key = os.getenv("USDA_API_KEY", "").strip()
    if not api_key:
        return {"error": "missing USDA_API_KEY"}

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

    return {
        "foods": foods,
        "count": len(foods),
        "total_hits": data.get("totalHits"),
        "source": "usda_food_data_central",
    }
