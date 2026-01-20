# nutrition_mcp_server.py
# MCP server for recipe and nutrition retrieval (local stdio transport)
#
# Environment variable required:
#   SPOONACULAR_API_KEY

from fastmcp import FastMCP
import os
import httpx

# Initialize MCP server
mcp = FastMCP("nutrition-mcp")


@mcp.tool()
def search_recipe(
    q: str,
    diet: str | None = None,
    intolerances: str | None = None,
    number: int = 3
) -> dict:
    """
    Search recipes by keyword and extract basic nutrition summaries.

    Args:
        q: Search keyword, e.g. "high protein chicken salad"
        diet: Optional dietary preference ("vegetarian", "keto", etc.)
        intolerances: Optional comma-separated food intolerances,
                       e.g. "peanut,shellfish"
        number: Number of recipes to return

    Returns:
        A dictionary with the following structure:
        {
            "results": [
                {
                    "id": int,
                    "title": str,
                    "image": str,
                    "calories": float | None,
                    "protein": float | None,
                    "carbs": float | None,
                    "fat": float | None,
                    "url": str
                }
            ]
        }
    """

    # Read API key from environment variables
    api_key = os.getenv("SPOONACULAR_API_KEY")
    if not api_key:
        return {"error": "missing SPOONACULAR_API_KEY"}

    # Spoonacular API endpoint (can be replaced with another provider)
    base_url = "https://api.spoonacular.com/recipes/complexSearch"

    # Query parameters
    params = {
        "query": q,
        "diet": diet,
        "intolerances": intolerances,
        "addRecipeNutrition": True,
        "number": number,
        "apiKey": api_key,
    }

    # Send HTTP request
    response = httpx.get(base_url, params=params, timeout=30)
    data = response.json()

    results = []

    for item in data.get("results", []):
        # Build a nutrient lookup table by name (lowercased)
        nutrients = {
            n["name"].lower(): n
            for n in item.get("nutrition", {}).get("nutrients", [])
        }

        # Helper function to safely extract nutrient values
        def get_value(name: str, default=None):
            nutrient = nutrients.get(name.lower())
            return nutrient.get("amount") if nutrient else default

        # Assemble simplified recipe output
        results.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "image": item.get("image"),
            "calories": get_value("calories"),
            "protein": get_value("protein"),
            "carbs": get_value("carbohydrates"),
            "fat": get_value("fat"),
            "url": (
                f"https://spoonacular.com/recipes/"
                f"{item.get('title', 'recipe').replace(' ', '-')}-"
                f"{item.get('id')}"
            ),
        })

    return {"results": results}


if __name__ == "__main__":
    # For local development, stdio transport is preferred.
    # (Recommended for local tools and small projects;
    # SSE is more suitable for shared or multi-user setups.)
    mcp.run(transport="stdio")
