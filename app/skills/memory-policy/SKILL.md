---
name: memory-policy
description: Use this skill when deciding whether user information belongs in long-term memory under /memories/users/<user_id>/*.md.
---

# Memory Policy

## Overview

This skill defines what the agent should treat as durable user memory and what should stay out of long-term memory.

Long-term memory paths stay under:
- `/memories/users/<user_id>/...`

This skill is about **memory write rules**, not about meal logs, health sync state, or other domain records.

## Phase 1 Core Files

Use these as the default long-term memory targets:

- `/memories/users/<user_id>/profile.md`
- `/memories/users/<user_id>/preferences.md`

Later expansion files may exist:

- `/memories/users/<user_id>/goals.md`
- `/memories/users/<user_id>/shopping.md`

In the current phase, prefer `profile.md` and `preferences.md` unless the user gives explicit durable information that clearly belongs in goals or shopping habits.

## File Responsibilities

### `profile.md`

Use for:
- Confirmed allergies or intolerances
- Confirmed long-term dietary constraints
- Stable cooking context that materially changes recommendations

Do not use for:
- Meal events
- Temporary requests
- Guessed health conditions
- Sensitive identity details unless explicitly provided and clearly needed

### `preferences.md`

Use for:
- Stable food likes/dislikes
- Stable cuisine preferences
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

When in doubt, do not write.

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
