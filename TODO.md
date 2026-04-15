# TODO

## Stabilize Nutrition Estimation Routing And Fallback Selection
- Goal: make nutrition estimation stable across common home-style dishes, branded foods, and restaurant/menu items by improving source routing, candidate filtering, and serving-basis handling.
- Current status:
  - Observability for the nutrition subagent is in place and we can now see main-agent -> subagent -> tool flow.
  - The `nutrition-specialist` prompt has been shortened to a high-level skeleton; most detailed behavior rules now live in [`app/skills/nutrition/nutrition-lookup/SKILL.md`](/Users/elvasmac/Documents/Projects/ai-agent/app/skills/nutrition/nutrition-lookup/SKILL.md).
  - We also added a thin USDA tool warning for composed-dish rows with unclear serving basis in [`app/tools/nutrition_http_tools.py`](/Users/elvasmac/Documents/Projects/ai-agent/app/tools/nutrition_http_tools.py).
- What we learned from regression:
  - `咖喱饭 / curry rice` can sometimes land in a reasonable range, but it is not fully stable yet.
  - `炒饭 / fried rice` is usually acceptable.
  - `意大利面 / pasta with meat sauce` is better than before, but can still degrade into a broad range instead of one solid estimate.
  - `三明治 / sandwich` is unstable because the agent sometimes forgets to include bread and returns unrealistically low carbs.
  - Packaged/branded items like `Quest bar` and `Trader Joe's Mandarin Orange Chicken` are better than before but still not consistently stable.
  - Restaurant/menu items like `Big Mac` and `California roll` still drift into wrong candidates or wrong serving interpretations.
  - English prompts are not a full fix: some dish cases look better in English, but branded/menu items can still fail badly, so the root problem is source/candidate selection rather than Chinese wording.
- Main problems to fix next:
  - Add stronger structured metadata from tools so the model can more reliably reject bad candidates instead of inferring everything from raw text.
  - Improve routing for branded/menu items so named chain items prefer brand-aware or restaurant-aware paths over generic recipes.
  - Harden serving-basis rules for `one serving / one sandwich / one burger / one roll / 一份 / 一个 / 一卷` so partial-size candidates cannot be used as full-serving answers.
  - Strengthen decomposition defaults for categories that require mandatory core components, especially sandwich -> bread and sushi roll -> full roll rather than single piece.
- Follow-up test set to keep rerunning:
  - Curry rice with chicken and carrots
  - Fried rice with egg and shrimp
  - Pasta with meat sauce
  - Sandwich with chicken breast, lettuce, and sauce
  - Trader Joe's Mandarin Orange Chicken
  - Quest Chocolate Chip Cookie Dough protein bar
  - Subway 6-inch turkey sandwich
  - Big Mac
  - Banana
  - California roll

## Explore YouTube Playlist Ingestion And YouTube RAG Later
- Goal: later add a YouTube ingestion path that can pull videos from a playlist and support a YouTube-oriented RAG workflow.
- Notes:
  - This is not the current priority.
  - Keep this as a future exploration item only after the nutrition fallback/routing work is stable.
