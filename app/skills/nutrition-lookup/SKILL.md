---
name: nutrition-lookup
description: Use this skill for calorie/macronutrient lookup and nutrition fact questions. Choose nutrition tools based on food type, then report a single final estimate with a source label.
---

# Nutrition Lookup

## Overview

This skill handles nutrition lookup requests and prepares reliable calorie/macronutrient estimates.

## Tool Use Rules

1. Always derive a clean `food_query` before calling tools.
- Extract the real thing the user wants nutrition for: the food, drink, product, dish, brand item, or menu item.
- Preserve whatever wording best identifies that real item for lookup quality.
- Remove surrounding filler only when it is clearly not part of the item being looked up.
- Build an English lookup query before every nutrition tool call.
- The current nutrition search tools are English-oriented tools. Their `query` input must be English, even when the user asks in Chinese or another language.
- Do not send Chinese or other non-English lookup text directly into nutrition tool calls.

2. Field boundaries:
- `user_text`: the original user sentence.
- `meal_description`: concise user-facing description of the meal when the user is asking about intake or logging.
- `food_query`: the lookup-oriented name of the real item you chose to search.
- `food_query_en`: the English lookup phrase you will actually use for English-oriented search tools.

3. Choose tools by food type and source fit:
- `usda_search_foods`: best for generic ingredients, basic foods, and common staples.
- `spoonacular_search_recipe`: best for composed dishes, named recipes, and home-style plated meals.
- `openfoodfacts_search_products`: best for packaged or branded foods.
- `tavily_search_nutrition`: best as a fallback for restaurant items, niche foods, or cases where the structured sources are a poor fit.
- For tool calls, use `food_query_en` rather than the original Chinese wording.
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
- If the user is only asking for nutrition facts, answer the nutrition question directly and do not proactively switch into meal logging.

6. If tools still do not return usable macros:
- Ask one concise clarification question (portion size, recipe style, or major ingredients), unless user explicitly asked to proceed with estimate.

7. Reporting format:
- Give kcal/protein/carbs/fat with units.
- Label the chosen source.
- Respond in the user's language when practical, even if the lookup query was converted to English.

## Query Normalization Examples

- "我晚上吃了意大利面，可以帮我记录吗？" -> `意大利面`
- "我晚上吃了意大利面，可以帮我记录吗？" -> `food_query=意大利面`, `food_query_en=spaghetti`
- "就是普通的番茄肉酱意大利面" -> `food_query=番茄肉酱意大利面`, `food_query_en=spaghetti bolognese`
- "I had chicken rice tonight, please log it" -> `food_query_en=chicken rice`
- "记录一下我吃的牛油果吐司和鸡蛋" -> `food_query=牛油果吐司 鸡蛋`, `food_query_en=avocado toast egg`
- "吃茶三千的芒果绿茶热量多少" -> keep the branded drink item together and use an English lookup such as `food_query_en=Chi Cha San Chen mango green tea`
