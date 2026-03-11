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

2. Call tools in this order:
- First: `usda_search_foods(query=<food_query>, page_size=5, page_number=1)`
- Second (if USDA has no usable macros): `spoonacular_search_recipe(query=<food_query>, number=5)`

3. A usable result must have all four core values:
- calories (kcal)
- protein (g)
- carbs (g)
- fat (g)

4. If tools still do not return usable macros:
- Ask one concise clarification question (portion size, recipe style, or major ingredients), unless user explicitly asked to proceed with estimate.

5. Reporting format:
- Give kcal/protein/carbs/fat with units.
- Label source (`USDA` or `Spoonacular`).

## Query Normalization Examples

- "我晚上吃了意大利面，可以帮我记录吗？" -> `意大利面`
- "就是普通的番茄肉酱意大利面" -> `番茄肉酱意大利面`
- "I had chicken rice tonight, please log it" -> `chicken rice`
- "记录一下我吃的牛油果吐司和鸡蛋" -> `牛油果吐司 鸡蛋`
