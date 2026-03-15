---
name: nutrition-lookup
description: Use this skill for calorie/macronutrient lookup and nutrition fact questions. Extract a clean food query first, then call USDA and Spoonacular in order, and report numbers with source labels.
---

# Nutrition Lookup

## Overview

This skill handles nutrition lookup requests and prepares reliable calorie/macronutrient estimates.

## Tool Use Rules

1. Always derive a clean `food_query` before calling tools.
- Keep food/dish names and key style words (for example: grilled, tomato sauce, bolognese).
- Remove intent wrapper text such as "I ate", "help me log", "can you record", and similar filler.
- If the user writes in Chinese, also derive `food_query_en` for tool lookup. Keep `food_query` as the user-language dish phrase.
- This extraction should happen in the agent plan before the tool call. Do not pass wrapper sentences into tools and expect the service layer to repair them.

2. Field boundaries:
- `user_text`: the original user sentence.
- `meal_description`: concise user-facing description of the meal; can stay in user language.
- `food_query`: lookup-oriented dish phrase with wrappers removed.
- `food_query_en`: English lookup phrase for USDA/Spoonacular when `food_query` is Chinese or mixed-language.

3. Call tools in this order:
- First: `usda_search_foods(query=<food_query>, page_size=5, page_number=1)`
- Second (if USDA has no usable macros): `spoonacular_search_recipe(query=<food_query>, number=5)`
- Third fallback: `tavily_search_nutrition(query=<food_query>)`
- Fourth fallback for packaged/branded foods: `openfoodfacts_search_products(query=<food_query>)`
- For Chinese input, prefer `food_query_en` for the actual tool call and keep `food_query` for display/debug context.
- For meal logging, pass both `food_query` and `food_query_en` into `prepare_meal_log(...)` so the draft pipeline can reuse them directly.

4. A usable result must have all four core values:
- calories (kcal)
- protein (g)
- carbs (g)
- fat (g)

5. If tools still do not return usable macros:
- Ask one concise clarification question (portion size, recipe style, or major ingredients), unless user explicitly asked to proceed with estimate.

6. Reporting format:
- Give kcal/protein/carbs/fat with units.
- Label source (`USDA`, `Spoonacular`, `Tavily`, `OpenFoodFacts`, or `Estimated`).
- Respond in the user's language when practical, even if the lookup query was converted to English.

## Query Normalization Examples

- "我晚上吃了意大利面，可以帮我记录吗？" -> `意大利面`
- "我晚上吃了意大利面，可以帮我记录吗？" -> `food_query=意大利面`, `food_query_en=spaghetti`
- "就是普通的番茄肉酱意大利面" -> `番茄肉酱意大利面`
- "I had chicken rice tonight, please log it" -> `chicken rice`
- "记录一下我吃的牛油果吐司和鸡蛋" -> `牛油果吐司 鸡蛋`
