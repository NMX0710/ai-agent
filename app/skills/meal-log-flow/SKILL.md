---
name: meal-log-flow
description: Use this skill when users ask to log or record meals into Apple Health. Follow prepare -> show estimate -> explicit confirm -> commit flow.
---

# Meal Log Flow

## Overview

This skill controls the meal logging workflow and write safety for Apple Health sync.

## Mandatory Flow

1. Detect logging intent.
- Triggers include "记录", "log this meal", "save to Apple Health", and equivalent intent.

2. Build a clean `food_query` from the user sentence.
- Keep only food/dish phrase for lookup quality.

3. Create draft first.
- Call `prepare_meal_log(...)` with the meal description you normalized for lookup.

4. Show estimate before write.
- Present kcal/protein/carbs/fat to user.
- Ask for explicit confirmation.

5. Commit only after explicit confirmation.
- Call `commit_meal_log(draft_id=<id>, user_id=<id>, confirmed=true)` only after a clear user confirmation action.

## Guardrails

- Never call `commit_meal_log` without explicit confirmation.
- Never skip the estimate preview step.
- If nutrition data quality is weak, tell user the estimate is approximate and ask one brief follow-up when needed.

## Example Decisions

- User: "我晚上吃了意大利面，可以帮我记录吗？"
  - Normalize lookup phrase: `意大利面`
  - Prepare draft
  - Show estimate + ask for confirm
  - Commit only on confirmation

- User: "帮我记到苹果健康里"
  - If food details missing, ask one concise clarifying question first.
