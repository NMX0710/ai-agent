---
name: nutrition-tools
description: Use nutrition and recipe tools (Spoonacular and USDA FoodData Central) for food lookup, calorie/macronutrient support, and recipe search when users ask about meals, calories, ingredients, or nutrition facts.
---

# Nutrition Tools Skill

Use this skill when users ask for:
- Calories or macros of a food/meal
- Nutrition facts verification
- Recipe search with nutrition context
- Ingredient substitution with macro awareness

## Workflow

1. Determine whether the user needs:
- Recipe discovery
- Food nutrition lookup
- Both recipe and nutrition context

2. Select tools directly:
- Use `spoonacular_search_recipe` for recipe and Spoonacular nutrition capabilities.
- Use `usda_search_foods` for USDA FoodData Central lookups.

3. If user intent is broad (for example “what should I eat”), recipe tools may be enough.

4. If user asks numeric nutrition values, call at least one nutrition-focused tool before answering.

5. Report numbers clearly with units and mention source tool names you used.

## Guardrails

- Do not invent nutrition numbers when tools returned no usable data.
- Ask a brief clarification question when quantity/unit is missing and materially affects calories/macros.
- Keep answers concise and practical for meal decisions.
