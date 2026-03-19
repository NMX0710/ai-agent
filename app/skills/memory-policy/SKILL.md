---
name: memory-policy
description: Use this skill when a user states or updates a durable personal fact, stable food preference, recurring eating habit, dietary restriction, long-term goal, or other cross-conversation information that may be worth remembering under /memories/users/<user_id>/*.md, even if the user does not explicitly ask you to save or remember it.
---

# Memory Policy

## Overview

This skill defines what the agent should treat as durable user memory, what should stay out of long-term memory, and how to update memory files conservatively but proactively.

Long-term memory paths stay under:
- `/memories/users/<user_id>/...`

This skill is about **memory write rules**, not about meal logs, health sync state, or other domain records.

Use this skill whenever the user reveals something that sounds likely to matter again in future conversations, especially:
- stable likes or dislikes
- recurring eating habits
- durable dietary restrictions
- long-term goals
- stable cooking constraints

The user does **not** need to say "remember this" for this skill to apply.

## Phase 1 Core Files

Use these as the default long-term memory targets:

- `/memories/users/<user_id>/profile.md`
- `/memories/users/<user_id>/preferences.md`

Later expansion files may exist:

- `/memories/users/<user_id>/goals.md`
- `/memories/users/<user_id>/shopping.md`

In the current phase, prefer `profile.md` and `preferences.md` unless the user gives explicit durable information that clearly belongs in goals or shopping habits.

For stable food preferences, recurring eating habits, ingredient avoidances, cuisine likes/dislikes, and similar dietary-preference memory, write to:
- `/memories/users/<user_id>/preferences.md`

In the current phase, do not invent new filenames for dietary preference memory.
Do not create files such as:
- `/memories/users/<user_id>/dietary_preferences.md`
- `/memories/users/default_user/preferences.md`
- any other free-form memory filename for preference tracking

Always use the current runtime `user_id` in the memory path. Never substitute placeholder ids such as `default_user`, `anonymous`, or guessed user ids.

## Trigger Rules

This skill should trigger when the user explicitly states information that is likely to remain true across sessions, including:
- "I usually skip breakfast."
- "I do not eat breakfast."
- "I avoid dairy."
- "I do not like cilantro."
- "I prefer quick meals on weekdays."
- "I am trying to eat more protein."

This skill should usually **not** trigger for:
- one-time cravings
- today-only plans
- single recipe requests
- temporary moods
- facts inferred only from behavior rather than stated by the user

Examples that should usually **not** be written:
- "I did not eat breakfast today."
- "Tonight I want something light."
- "This time do not add onions."
- "Can you log this meal?"

Statements such as "I usually ...", "I normally ...", "I do not eat ...", "I always avoid ...", "我平常...", "我一般...", and "我不吃..." are strong signals of stable preference or recurring habit. For these cases, do not ask a redundant confirmation question before writing long-term memory unless the user also signals uncertainty or temporariness.

## Default Workflow

When this skill triggers:

1. Decide whether the user stated a durable fact, stable preference, recurring habit, or long-term goal.
2. Decide the correct target file:
- `profile.md` for durable constraints or stable user facts that materially affect recommendations.
- `preferences.md` for likes, dislikes, cuisine preferences, recurring meal habits, budget/time preferences, and ingredient avoidances.
- `goals.md` only for explicit long-term nutrition or lifestyle goals.
- `shopping.md` only for explicit stable brand/store habits.
3. Use the current runtime `user_id` when constructing the path.
4. Read the existing target memory file before updating it.
5. Write a short durable summary only if the new information is explicit and likely to matter again.
6. Update the canonical file in place instead of inventing a new memory file or appending noisy session-by-session notes.

If the statement is ambiguous between temporary and durable, do not write it yet.

## Preferences File Format

Use a simple bullet-list structure in `/memories/users/<user_id>/preferences.md`.

Preferred format:

```md
- Usually skips breakfast.
- Avoids cilantro.
- Prefers quick weekday meals.
```

Formatting rules:
- one durable preference or recurring habit per bullet
- short plain-language summaries
- prefer stable summaries over chat-specific wording
- do not write loose paragraphs when a bullet summary is sufficient
- update an existing bullet when the same preference is clarified later
- if the user speaks Chinese, you may write the memory in Chinese, but keep the same bullet-list structure

Good examples:
- `- 平时通常不吃早餐。`
- `- Usually does not eat breakfast.`
- `- Avoids dairy.`

Avoid formats like:
- raw unlabeled paragraphs
- session transcripts
- duplicated bullets for the same preference
- filenames other than `preferences.md` for preference memory

## File Responsibilities

### `profile.md`

Use for:
- Confirmed allergies or intolerances
- Confirmed long-term dietary constraints
- Stable cooking context that materially changes recommendations
- Stable user facts that materially change safe recommendations

Do not use for:
- Meal events
- Temporary requests
- Guessed health conditions
- Sensitive identity details unless explicitly provided and clearly needed

### `preferences.md`

Use for:
- Stable food likes/dislikes
- Stable cuisine preferences
- Recurring meal habits or meal-pattern preferences
- Stable budget or cooking-time preferences
- Repeated ingredient preferences or avoidances

Do not use for:
- One-time cravings
- Single-session recipe requests
- Temporary exceptions

### `goals.md`

Reserve for:
- Explicit long-term nutrition or lifestyle goals

Do not use for:
- Single-day plans
- Agent-inferred goals

### `shopping.md`

Reserve for:
- Explicit stable store, brand, or purchasing habits

Do not use for:
- Shopping carts
- Price snapshots
- Inventory state

## Conservative Auto-Write Policy

Auto-write is appropriate only for:
- Explicit allergies or intolerances
- Explicit long-term diet goals
- Explicit and stable preferences
- Explicit and stable shopping habits

Do not auto-write:
- Inferred health judgments
- Sensitive personal information
- One-time comments
- Temporary plans or moods
- Information the agent is only guessing from context

When in doubt, do not write, but do not set the bar so high that clearly stated stable preferences are ignored.

## What Is Not Long-Term Memory

Do not store these as long-term memory markdown records:

- Meal logs
- Weight history
- Apple Health sync state
- Packaged-food records or future food catalog entries

These are domain records. Long-term memory may contain only a high-level stable summary derived from repeated behavior, never the detailed record stream itself.

## Update Rules

- Read existing memory before writing new memory
- If a new user statement clearly updates a stable fact, revise the existing memory entry
- If the information is ambiguous, temporary, or only useful for the current chat, do not write it
- Prefer short, durable summaries over append-only notes
- Prefer plain-English summaries that will still make sense in a later session

## Concrete Classification Examples

- User: "I usually skip breakfast."
  - Write to: `preferences.md`
  - Example entry: `- Usually skips breakfast.`
  - Reason: recurring eating habit that may affect future meal planning

- User: "I do not eat breakfast."
  - Write to: `preferences.md`
  - Example entry: `- Usually does not eat breakfast.`
  - Reason: stable meal-pattern preference unless later corrected

- User: "我平常不吃早餐"
  - Write to: `/memories/users/<user_id>/preferences.md`
  - Example entry: `- 平时通常不吃早餐。`
  - Do not create: `dietary_preferences.md`
  - Reason: recurring meal-pattern preference belongs in the canonical preferences file

- User: "I am lactose intolerant."
  - Write to: `profile.md`
  - Reason: durable dietary constraint with safety implications

- User: "I prefer meals that take less than 20 minutes on weekdays."
  - Write to: `preferences.md`
  - Reason: stable cooking-time preference

- User: "I want to lose 10 pounds this month."
  - Consider: `goals.md`
  - Reason: explicit longer-term goal, if goals are being used in the current phase

- User: "I skipped breakfast today."
  - Do not write
  - Reason: single-day event, not a stable preference

- User: "Tonight I want a low-carb dinner."
  - Do not write
  - Reason: session-local request, not a durable preference by itself
