---
name: meal-log-flow
description: Use this skill when users ask to log or record meals into Apple Health. Follow lookup -> choose final estimate -> prepare -> show estimate -> explicit confirm -> commit flow.
---

# Meal Log Flow

## Overview

This skill controls the meal logging workflow and write safety for Apple Health sync.

## Mandatory Flow

1. Detect logging intent.
- Triggers include "记录", "log this meal", "save to Apple Health", and equivalent intent.

2. Build a clean `food_query` from the user sentence.
- Keep only food/dish phrase for lookup quality.
- If user input is Chinese, also derive `food_query_en` for USDA/Spoonacular lookup.
- This is the agent's job.

3. Look up nutrition before drafting.
- Choose the nutrition tool or tools that best fit the food type.
- Decide on one final estimate for kcal/protein/carbs/fat and a source label.
- `prepare_meal_log(...)` does not do nutrition lookup for you.

4. Create draft after you have the final estimate.
- Call `prepare_meal_log(...)` with:
- `meal_description`: concise user-facing description of what the user ate.
- `energy_kcal`, `protein_g`, `carbs_g`, `fat_g`: the final estimate you chose.
- `nutrition_source`: the source you used for the final estimate.
- `nutrition_confidence`: optional when you have a meaningful confidence judgment.
- If the user already asked to log the meal and the food is clear enough, create the draft immediately only after you have a usable final estimate. Do not ask a redundant question like “Do you want me to generate a draft?” first.
- If you do not yet have a usable final estimate, do not call `prepare_meal_log(...)`. Ask one concise clarification question or explicitly choose an approximate estimate first.

5. Show estimate before write.
- Present kcal/protein/carbs/fat to user.
- Ask for explicit confirmation.

6. Commit only after explicit confirmation.
- Call `commit_meal_log(draft_id=<id>, user_id=<id>, confirmed=true)` only after a clear user confirmation action.

## Guardrails

- Never call `commit_meal_log` without explicit confirmation.
- Never skip the estimate preview step.
- If nutrition data quality is weak, tell user the estimate is approximate and ask one brief follow-up when needed.

## Example Decisions

- User: "我晚上吃了意大利面，可以帮我记录吗？"
  - meal_description: `晚餐吃了意大利面`
  - choose nutrition lookup tool(s) and one final estimate
  - Prepare draft immediately after selecting a usable final estimate
  - Show estimate + ask for confirm
  - Commit only on confirmation

- User: "帮我记到苹果健康里"
  - If food details missing, ask one concise clarifying question first.
