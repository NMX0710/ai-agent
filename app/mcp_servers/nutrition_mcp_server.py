# nutrition_mcp_server.py
# MCP 服务：食谱/营养检索（本地 stdio 传输）
# 环境变量：SPOONACULAR_API_KEY
from fastmcp import FastMCP
import os, httpx

mcp = FastMCP("nutrition-mcp")

@mcp.tool()
def search_recipe(q: str, diet: str | None = None, intolerances: str | None = None, number: int = 3) -> dict:
    """
    按关键词检索食谱，并抽取热量等营养摘要。
    参数:
      q: 关键词，如 "high protein chicken salad"
      diet: 可选饮食模式 ("vegetarian", "keto"...)
      intolerances: 过敏原逗号分隔，如 "peanut,shellfish"
      number: 返回条数
    返回字段:
      results: [{id,title,image,calories,protein,carbs,fat,url}]
    """
    api_key = os.getenv("SPOONACULAR_API_KEY")

    if not api_key:
        return {"error": "missing SPOONACULAR_API_KEY"}

    # 示例：Spoonacular（可替换为你的 API）
    base = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": q, "diet": diet, "intolerances": intolerances,
        "addRecipeNutrition": True, "number": number, "apiKey": api_key
    }
    r = httpx.get(base, params=params, timeout=30)
    data = r.json()
    results = []
    for it in data.get("results", []):
        # 简化提取：第一项一般是 calories，后续粗分三大营养素
        nutrients = {n["name"].lower(): n for n in it.get("nutrition", {}).get("nutrients", [])}
        def val(name, default=None):
            n = nutrients.get(name.lower())
            return n.get("amount") if n else default
        results.append({
            "id": it.get("id"),
            "title": it.get("title"),
            "image": it.get("image"),
            "calories": val("calories"),
            "protein": val("protein"),
            "carbs": val("carbohydrates"),
            "fat": val("fat"),
            "url": f'https://spoonacular.com/recipes/{it.get("title","recipe").replace(" ","-")}-{it.get("id")}'
        })
    return {"results": results}

if __name__ == "__main__":
    # 本地开发优先 stdio（鱼皮也推荐：本地/小项目优先 stdio；SSE 适合多人共享）:contentReference[oaicite:1]{index=1}
    mcp.run(transport="stdio")
