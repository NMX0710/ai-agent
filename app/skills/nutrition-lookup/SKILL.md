---
name: nutrition-lookup
description: Use this skill for calorie/macronutrient lookup and nutrition fact questions. Choose nutrition tools based on food type, then report a single final estimate with a source label.
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

3. Choose tools by food type and source fit:
- `usda_search_foods`: best for generic ingredients, basic foods, and common staples.
- `spoonacular_search_recipe`: best for composed dishes, named recipes, and home-style plated meals.
- `openfoodfacts_search_products`: best for packaged or branded foods.
- `tavily_search_nutrition`: best as a fallback for restaurant items, niche foods, or cases where the structured sources are a poor fit.
- For Chinese input, prefer `food_query_en` when a tool works better with English dish names.
- Use as few tools as needed. Stop once you have a credible estimate from a well-matched source.

4. A usable result must have all four core values:
- calories (kcal)
- protein (g)
- carbs (g)
- fat (g)

5. After lookup:
- Pick one final estimate to present. Do not dump multiple conflicting tool outputs without choosing.
- Label the source you actually used (`USDA`, `Spoonacular`, `Tavily`, `OpenFoodFacts`, or `Estimated`).
- If the user is asking to log the meal, pass the final chosen estimate into `prepare_meal_log(...)`.

6. If tools still do not return usable macros:
- Ask one concise clarification question (portion size, recipe style, or major ingredients), unless user explicitly asked to proceed with estimate.

7. Reporting format:
- Give kcal/protein/carbs/fat with units.
- Label the chosen source.
- Respond in the user's language when practical, even if the lookup query was converted to English.

## Query Normalization Examples

- "我晚上吃了意大利面，可以帮我记录吗？" -> `意大利面`
- "我晚上吃了意大利面，可以帮我记录吗？" -> `food_query=意大利面`, `food_query_en=spaghetti`
- "就是普通的番茄肉酱意大利面" -> `番茄肉酱意大利面`
- "I had chicken rice tonight, please log it" -> `chicken rice`
- "记录一下我吃的牛油果吐司和鸡蛋" -> `牛油果吐司 鸡蛋`
