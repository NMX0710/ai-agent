---
name: nutrition-lookup
description: Use this skill for calorie and macro estimation with reliability checks. Before choosing a final estimate, check serving basis and portion level, prefer recipe-style lookup for composed dishes before decomposition, use compact recipe-name queries for Spoonacular, distinguish whole-meal requests from per-100g or small-serving database entries, semantically validate ingredient candidates during fallback, ask at most one key clarification question when needed, and avoid dumping raw provider candidates.
---

# Nutrition Lookup

## Overview

This skill is for the nutrition specialist subagent. Its job is to turn a user food description into one usable nutrition estimate or one concise clarification question.

Do not answer from general nutrition knowledge alone when a nutrition lookup tool fits the request. Use the tools first, then reason over the returned candidates.

## Core Workflow

1. Identify the actual food, drink, product, or dish the user wants nutrition for.
2. Convert the lookup phrase into English before calling nutrition tools.
3. If the request is a composed dish, named meal, or whole plate, try recipe-style lookup first.
4. Choose the best-matched nutrition tool for the food type.
5. Inspect the returned candidates and decide whether the result is good enough, semantically matched, and at the right serving basis.
6. If direct lookup is weak but the dish is recognizable, estimate a common single serving by decomposing the dish into ingredients.
7. If the estimate is credible, choose one final estimate and label the source.
8. If it is still not good enough, ask one concise clarification question about the highest-value missing detail.

## Tool Priority For Dishes

When the user asks about a composed dish, named recipe, plated meal, bowl, or plate:

1. Start with `spoonacular_search_recipe`.
2. Use a short recipe-style English dish name for Spoonacular, not a long ingredient-stuffed phrase.
3. Keep the Spoonacular query close to the canonical dish name such as `chicken curry rice`, `chicken curry`, or `japanese chicken curry` instead of concatenating every known ingredient.
4. Use user-provided ingredients mainly to validate or rank recipe candidates, or to guide fallback decomposition if recipe lookup fails.
5. Use a recipe-style result directly if it is a good semantic match and looks like a plausible serving-level estimate.
6. Only fall back to ingredient decomposition if recipe-style lookup:
- returns no useful result
- returns a clearly mismatched dish
- returns only low-granularity or unclear-basis results
- returns an implausible quantity for the user's described portion

Do not default to ingredient decomposition for every dish-level request.

## Recipe Candidate Matching

Finding a Spoonacular result is not enough by itself. Check whether the candidate still looks like the same kind of dish the user asked about.

Use a Spoonacular recipe only when most of the following are true:
- the title is a close match to the requested dish name
- the dish form matches, such as rice dish vs soup vs curry-only entree
- the user-provided ingredients fit naturally into the candidate
- the kcal and macros look plausible for a typical serving of that dish

Reject or down-rank recipe candidates when:
- the title drifts into a different dish type such as `soup`
- the recipe appears to be a substantially different style or format from the user's dish
- the macro profile is obviously inconsistent with the expected dish, such as extremely low protein for a chicken-and-rice meal or unusually high fat for a plain home-style curry rice
- for a chicken-and-rice curry meal, a candidate with very low protein or very low carbs for a full serving is usually a weak match rather than a reliable final answer

If the best Spoonacular candidates are only weak matches, do not force one of them. Fall back to decomposition.

For example, if Spoonacular returns a recipe like `Creamy Curry Chicken With Yellow Rice` with about 335 kcal but only about 3 to 4 g protein for a full serving, treat that as a weak match for ordinary chicken curry rice and do not use it as the final answer.

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
- For a full plated meal, burger, sandwich, pasta plate, curry rice serving, or sushi roll, totals around 100 to 150 kcal are usually not credible final answers unless the user explicitly described a tiny portion.

5. Does the candidate semantically match the intended food?
- Do not use a result that materially changes the food identity.
- For recipe-style lookup, reject candidates whose title or dish form drifts too far from the requested dish.
- For ingredient fallback, reject candidates that add irrelevant preparation or dish-level modifiers when the user asked for a basic ingredient.
- Examples to reject for a basic ingredient query include descriptions like `breaded`, `battered`, `fried`, `microwaved`, `with oil`, or unrelated dish names.
- If the top candidate is mismatched, inspect another candidate, refine the query, or switch lookup path.

## Tool Metadata Interpretation

Some nutrition tools return structured guardrail fields. Treat them as strong evidence, not optional decoration.

Important metadata fields include:
- `is_composed_dish`
- `serving_basis_unclear`
- `likely_per_100g_or_small_portion`
- `not_recommended_for_full_serving_estimate`
- `nutrition_basis`
- `brand_match_confident`
- `exact_name_match`
- `dish_form_match`
- `name_match_score`
- `restaurant_query_mismatch`
- `candidate_warnings`
- `internal_consistency_warning`
- `query_context`

How to use them:
- If a candidate has `not_recommended_for_full_serving_estimate = true`, do not use it as the final answer for a whole dish, full roll, one sandwich, one burger, one bar, or other full-serving request.
- If `nutrition_basis` is `per_100g_fallback`, do not pretend it is automatically the nutrition for one packaged item. Use it only when the serving interpretation is still credible.
- If `brand_match_confident` is false for a branded query, treat the row as a weak candidate unless no better branded match exists.
- If `exact_name_match` is true for a branded packaged item and serving-level data is present, prefer that candidate over looser branded matches.
- If `dish_form_match` is false for a recipe candidate, reject it for whole-dish estimation.
- If `restaurant_query_mismatch` is true, do not use that row as the final answer for a named restaurant or chain menu item.
- If `candidate_warnings` or `internal_consistency_warning` indicate that calories and macros are incoherent, reject that candidate and switch sources or fall back.
- Read `query_context` to understand whether the tool is already flagging the request as a full-serving dish, restaurant item, branded product, or a case with mandatory components.

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
- For common home-style dishes like chicken curry rice, if the user already named the main ingredients and did not ask for precision, prefer a common single-serving estimate over a follow-up question.

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
- If the user says `一份`, `一个`, `一卷`, `one sandwich`, `one burger`, `one roll`, or similar portion-level wording, reject candidates that still look like per-100 g, side-size, or partial-component entries.

## Restaurant And Brand Policy

When the user asks about a chain menu item, restaurant dish, or branded prepared meal:

- Prefer a brand-aware or restaurant-aware lookup path over a generic homemade recipe approximation.
- For packaged branded items, Open Food Facts or branded database rows are often better than generic recipe search.
- For chain menu items, weak generic recipe matches should not override the clearly named brand or menu item.
- If calories and macros are internally inconsistent for the named branded item, reject that candidate and try a more brand-specific path.
- If a chain-menu or branded candidate lacks a confident brand match, treat it as suspect even if the food name looks roughly similar.
- For a packaged branded item like a protein bar or frozen entree, prefer one serving-level branded value over a broad range assembled from mixed generic candidates.
- When Open Food Facts returns several near-duplicate branded hits for the same exact item, choose the strongest serving-level exact-name match instead of returning a range.

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
- For a request like chicken curry rice with carrots, if portion size is missing but no exact logging precision was requested, default to a common single serving instead of asking for grams.

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
4. Plan the smallest useful ingredient set before calling tools so you do not search unnecessary extras.
5. Estimate a typical single serving for each component.
6. Sum kcal, protein, carbs, and fat across the components.
7. Tell the user that this is an approximate estimate based on a common serving.

Examples of common components:
- chicken curry rice: rice, chicken, carrots, curry sauce or roux, cooking oil
- fried rice: rice, oil, egg, aromatics, common protein or vegetables
- sandwich: bread, filling protein, sauce, cheese or vegetables when typical
- California roll: sushi rice, imitation crab or crab mix, avocado, cucumber, nori

Do not pretend the decomposition is exact. It is a fallback for a more credible meal-level estimate when direct lookup is weak.

## Ingredient Candidate Matching

When decomposition requires ingredient lookup, do not treat the first returned ingredient candidate as automatically acceptable.

Check whether the candidate still represents the same ingredient the user intended:
- `chicken breast cooked` should not become breaded tenders or nuggets
- `cooking oil` should not become a cooked dish that merely contains oil
- `carrot` should not become a mixed prepared side dish

If the candidate drifts into a different food identity, reject it and try another candidate or a cleaner query.

## Mandatory Component Sanity Checks

Some dish categories require core components unless the user explicitly says otherwise.

- Sandwich: include bread by default. If the user asked about a normal sandwich and your estimate ends up with almost no carbs, your estimate is probably missing bread and should not be returned yet.
- Burger: include bun and patty by default for a normal burger or named burger item.
- Sushi roll: treat `one roll`, `一卷`, or `一份加州卷` as a full roll, not a single piece. A near-100-kcal answer is usually too small unless the user explicitly asked about one piece.
- Pasta plate: for a standard pasta dish, include the pasta itself as a major carb component; do not let sauce-only reasoning become the final answer.

If a mandatory core component is missing, revise the decomposition or reject the candidate.

## Tool Selection

- `usda_search_foods`: generic ingredients and simple foods
- `spoonacular_search_recipe`: composed dishes and named meals
- `openfoodfacts_search_products`: packaged or branded foods
- `tavily_search_nutrition`: fallback for restaurant items or poor fits for structured sources

Use as few tools as needed. Stop once you have one credible estimate.

For composed dishes and plated meals:
- Prefer `spoonacular_search_recipe` first.
- If recipe-style lookup fails, do not treat `usda_search_foods` as a second recipe database for the whole dish name.
- After recipe-style lookup fails, use `usda_search_foods` mainly for ingredient-level decomposition, not for generic whole-dish rows like `chicken curry with rice`.
- If Spoonacular returns one canonical match and one style-specific variant, prefer the canonical match rather than returning a range by default.

For restaurant and chain menu items:
- Use `tavily_search_nutrition` as a higher-value fallback than a generic homemade recipe when the user names a restaurant or menu item.
- Do not answer a branded chain item like `Big Mac` with a generic burger estimate that clearly does not match known serving-level nutrition.
- For restaurant items, prefer official brand or restaurant pages when Tavily surfaces them. Treat generic article snippets as weaker evidence.

## Final Estimate Rule

- Do not dump multiple conflicting candidates to the user.
- Pick one final estimate whenever the available evidence is good enough.
- A final estimate should include kcal, protein, carbs, and fat when available.
- Label the source you actually used.
- If the final estimate comes from decomposition or a default serving assumption, say that clearly.
- Do not return a range when one common serving estimate is already good enough for the user's request.
- If several plausible candidates cluster in a similar band, choose one representative serving-level estimate instead of surfacing the whole band.

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

Bad fallback behavior:
- Spoonacular returns no useful recipe result
- USDA returns a generic row such as `Chicken curry with rice` at 116 kcal
- Return that directly as the whole-dish answer

Why it is still bad:
- This is still likely a low-granularity or small-serving entry rather than a full plate
- A chicken curry rice meal with rice, chicken, and sauce is usually several hundred kcal, not close to 100 kcal
- For a typical single serving of chicken curry rice, values roughly in the 400 to 600 kcal range are often more credible than ~100 kcal

Better fallback after recipe search fails:
- Do not use the low-calorie generic whole-dish USDA row as the final answer
- Decompose into rice, chicken, carrots, curry sauce or roux, and cooking oil
- Or ask one concise clarification question if portion size is too uncertain

Preferred behavior for this case:
- For `咖喱饭` with chicken and carrots, when the user asks for a normal estimate and does not provide grams, assume a common single serving
- Estimate from a typical home-style serving and clearly label it as approximate
- Do not block on a grams question unless the user explicitly wants high precision
- If Spoonacular only returns weak curry-like recipes with implausible macros for chicken curry rice, reject them and move to decomposition

## Sushi Roll Example

User: "一份加州卷寿司的热量大概是多少？"

Bad behavior:
- Return about 94 kcal as the answer for one full California roll

Why it is bad:
- The user asked about a full roll, not a single bite or a tiny partial serving
- A California roll usually contains rice, filling, and nori across multiple pieces
- A value near 100 kcal is likely too small for a full roll serving
- For a typical full California roll serving, a result in the several-hundred-kcal range is usually more credible than ~100 kcal

Better behavior:
- Treat `一份` or `one roll` as a full-roll portion
- Reject small or unclear-basis entries and use a plausible full-roll estimate instead
- If no strong source is available, prefer a common full-roll estimate over a tiny partial-serving value
- If decomposition is needed, estimate from sushi rice, crab mix, avocado, cucumber, and nori for one standard roll
- For a typical California roll, a rough estimate in the low hundreds of kcal is usually more credible than a double-digit or near-100-kcal answer

## Fast Food Example

User: "一个巨无霸 Big Mac 的热量大概是多少？"

Bad behavior:
- Return a very high calorie number together with very low protein and carbs that do not make internal sense
- Use a generic burger-style candidate that does not clearly match the named menu item

Better behavior:
- Recognize `Big Mac` as a specific chain menu item
- Prefer a restaurant-aware or brand-aware lookup path
- Reject candidates whose calories and macros are internally inconsistent for a Big Mac

## Restaurant Sandwich Example

User: "Subway 6-inch turkey sandwich 的热量是多少？"

Bad behavior:
- Return a generic low-calorie turkey sandwich row that is not clearly Subway-specific

Better behavior:
- Treat `Subway` as a named restaurant/menu context
- Reject generic sandwich candidates if the tool metadata indicates a restaurant mismatch
- Prefer a restaurant-aware value or a common 6-inch Subway-sized estimate over a small generic sandwich row

## Branded Product Example

User: "Quest Chocolate Chip Cookie Dough protein bar 的热量是多少？"

Bad behavior:
- Return a broad range assembled from mixed generic protein bar candidates
- Use per-100 g fallback values as if they were definitely for one bar

Better behavior:
- Prefer a branded packaged-food result with a confident Quest match
- Prefer serving-level product nutrition over per-100 g fallback
- If Open Food Facts returns multiple near-duplicate Quest matches at one serving each, choose the strongest exact-name serving-level row instead of returning a range
- If the best branded row is only per-100 g with unclear serving size, ask one concise clarification question or state a cautious approximate assumption instead of pretending the value is exact

## Logging Context

If the upstream request is for meal logging, your job is still to return one final estimate or one concise clarification question. Do not call meal log tools from this subagent.
