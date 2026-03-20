---
name: nutrition-lookup
description: Use this skill for calorie and macro estimation with reliability checks. Before choosing a final estimate, check serving basis and portion level, distinguish whole-meal requests from per-100g or small-serving database entries, decompose dishes into common ingredients when direct dish lookup is missing or unreliable, ask at most one key clarification question when needed, and avoid dumping raw provider candidates.
---

# Nutrition Lookup

## Overview

This skill is for the nutrition specialist subagent. Its job is to turn a user food description into one usable nutrition estimate or one concise clarification question.

## Core Workflow

1. Identify the actual food, drink, product, or dish the user wants nutrition for.
2. Convert the lookup phrase into English before calling nutrition tools.
3. Choose the best-matched nutrition tool for the food type.
4. Inspect the returned candidates and decide whether the result is good enough and at the right serving basis.
5. If direct lookup is weak but the dish is recognizable, estimate a common single serving by decomposing the dish into ingredients.
6. If the estimate is credible, choose one final estimate and label the source.
7. If it is still not good enough, ask one concise clarification question about the highest-value missing detail.

## Estimate Reliability Check

Before returning a final estimate, check all of the following:

1. What level did the user ask about?
- ingredient
- packaged product
- single food item
- whole dish or whole meal

2. What level does the candidate result represent?
- likely per 100 g
- likely small serving
- likely generic database entry
- likely full serving or plated meal

3. Do the user request and the candidate result match?
- If the user asked about a whole dish or whole meal, do not directly return a likely per-100 g, small-serving, or generic low-granularity value as the final answer.

4. Does the quantity pass a sanity check?
- If the estimate is implausibly low or high for the described dish, do not return it as-is.
- Reassess the serving basis, try a better lookup path, decompose the dish, or ask one concise clarification question.

## Ambiguity Handling

When the food is too broad or could map to very different meals, do not dump several raw candidates back to the user.

Examples of ambiguous foods:
- curry rice / 咖喱饭
- fried rice / 炒饭
- pasta / 意大利面
- sandwich / 三明治

For these cases:
- Prefer one concise clarification question about the main protein, major ingredients, or portion size.
- If the user clearly wants an estimate without further detail, or if a follow-up would be unhelpful, choose a common default version of the dish and explicitly mark it as approximate.

## Whole Meal vs Database Entry

Treat user descriptions such as these as whole-dish or meal-level requests unless the user says otherwise:
- 咖喱饭
- 炒饭
- 盖饭
- pasta
- sandwich
- lunch
- dinner
- one bowl
- one plate

For those requests:
- Do not assume a generic database row already represents the user's whole portion.
- Watch for entries that look like ingredient-level, per-100 g, or low-granularity generic foods.
- If the tool result basis is unclear, prefer clarification or approximate serving logic over falsely precise final numbers.

## Clarification Policy

- Ask at most one concise, high-value question.
- Ask only when the answer would materially change the estimate.
- Good clarification targets:
  - main protein or main ingredients
  - portion size
  - restaurant vs home-style version

Example:
- User: "我晚饭吃了咖喱饭 帮我记录热量"
- Good question: "是鸡肉咖喱饭、牛肉咖喱饭，还是蔬菜咖喱饭？大概一碗还是一盘？"

## Approximate Estimate Policy

If the user does not provide more detail, you may estimate using a common version of the dish.

When you do this:
- Say that the estimate is approximate.
- State the default assumption briefly.
- Still choose one final estimate instead of showing a menu of provider candidates.

Example:
- "我先按常见鸡肉咖喱饭、一碗普通份量来估算。"

## Dish Decomposition Fallback

If direct dish lookup is missing, low-granularity, or not credible for the user's portion, estimate by decomposing the dish into common ingredients.

Use this fallback especially when:
- recipe search returns no useful result
- structured databases return only generic or unclear-basis entries
- the returned quantity is obviously implausible for the described dish
- the dish is common enough that its main components are easy to infer

Decomposition workflow:
1. Identify the dish's core components.
2. Replace defaults with any ingredients the user explicitly provided.
3. Add common base components that are normally present.
4. Estimate a typical single serving for each component.
5. Sum kcal, protein, carbs, and fat across the components.
6. Tell the user that this is an approximate estimate based on a common serving.

Examples of common components:
- chicken curry rice: rice, chicken, carrots, curry sauce or roux, cooking oil
- fried rice: rice, oil, egg, aromatics, common protein or vegetables
- sandwich: bread, filling protein, sauce, cheese or vegetables when typical

Do not pretend the decomposition is exact. It is a fallback for a more credible meal-level estimate when direct lookup is weak.

## Tool Selection

- `usda_search_foods`: generic ingredients and simple foods
- `spoonacular_search_recipe`: composed dishes and named meals
- `openfoodfacts_search_products`: packaged or branded foods
- `tavily_search_nutrition`: fallback for restaurant items or poor fits for structured sources

Use as few tools as needed. Stop once you have one credible estimate.

## Final Estimate Rule

- Do not dump multiple conflicting candidates to the user.
- Pick one final estimate whenever the available evidence is good enough.
- A final estimate should include kcal, protein, carbs, and fat when available.
- Label the source you actually used.
- If the final estimate comes from decomposition or a default serving assumption, say that clearly.

## Reliability Example

User: "咖喱饭的热量是多少？我放了胡萝卜和鸡肉"

Bad behavior:
- Search a generic database entry like "Chicken curry with rice"
- See a low calorie value such as 116 kcal
- Return that directly as the whole-meal answer

Why it is bad:
- The user asked about a whole dish
- The candidate is likely low-granularity or based on a small serving or 100 g-like basis
- The quantity is implausibly low for a normal serving of chicken curry rice

Better behavior:
- Ask one concise portion question, such as whether it was a bowl or plate
- Or estimate a common serving by using chicken, carrots, rice, curry sauce, and oil
- Clearly say that the result is approximate and based on a common serving

## Logging Context

If the upstream request is for meal logging, your job is still to return one final estimate or one concise clarification question. Do not call meal log tools from this subagent.
