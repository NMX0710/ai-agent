# Long-Term Memory Design Note

## Scope

This note defines the **active long-term memory shape** for the current Deep Agent runtime.

- Agent-facing memory paths stay under `/memories/users/<user_id>/...`
- The agent continues to read and write markdown-like memory files
- Future persistence should replace the **store adapter** under `StoreBackend`, not reintroduce the old `app/memory/` subsystem

This note does **not** define a new generic memory engine.

## Current Runtime Contract

The active runtime currently has two memory layers:

- Short-term memory: `MemorySaver()` with `thread_id = user_id:chat_id`
- Long-term memory semantics: `/memories/...` routed through `StoreBackend`

Today, the `/memories/...` layer is backed by `InMemoryStore()`, so it behaves like files from the agent's point of view, but it is not yet durable across process restarts.

## Separate The Two Concepts

There are two separate design questions:

1. **Memory files**
- These are the target files the agent may read and update under `/memories/users/<user_id>/`

2. **Memory write rules**
- These define when the agent may write, what it may write automatically, and what must stay out of long-term memory

These should stay separate. A memory file is the storage target. The write rules control agent behavior.

## File Priority

### Phase 1 Core

- `profile.md`
- `preferences.md`

These are the smallest useful long-term memory files already aligned with the current prompt shape.

### Later Expansion

- `goals.md`
- `shopping.md`

These are useful, but they should be added only after the core file boundaries and write rules are stable.

## Memory Files

### `profile.md`

Purpose:
- Stable user background that materially affects recommendations

Should contain:
- Confirmed allergies or intolerances
- Confirmed dietary constraints
- Stable cooking context that changes advice quality
- Stable language or cultural food context when explicitly stated

Should not contain:
- Single-meal events
- Temporary requests
- Inferred medical conclusions
- Sensitive identity details unless the user explicitly provided them and they are clearly needed for dietary guidance

Auto-write allowed:
- Explicit allergies or intolerances
- Explicit long-term dietary constraints
- Explicit stable cooking constraints

Auto-write not allowed:
- Guessed health conditions
- Sensitive personal attributes inferred from chat
- One-off statements that are not clearly durable

Simple format suggestion:
- Short bullet list grouped by category

### `preferences.md`

Purpose:
- Stable taste, ingredient, cuisine, budget, and cooking-style preferences

Should contain:
- Foods the user consistently likes or avoids
- Preferred cuisines or meal styles
- Stable time/budget preferences
- Repeatedly expressed ingredient preferences

Should not contain:
- One-time cravings
- Single-session meal requests
- Temporary exceptions

Auto-write allowed:
- Explicit stable preferences
- Repeated preferences seen across chats
- Explicit avoidances that are not medical restrictions

Auto-write not allowed:
- Weak or ambiguous preferences
- One-off comments without evidence of stability

Simple format suggestion:
- Short bullet list by preference type

### `goals.md`

Purpose:
- Stable nutrition or lifestyle goals that guide future recommendations

Should contain:
- Explicit fat-loss, muscle-gain, maintenance, or protein goals
- Explicit longer-term meal planning goals

Should not contain:
- Single-day plans
- Temporary motivation
- Agent-inferred goals not stated by the user

Auto-write allowed:
- Explicit, ongoing goals stated by the user

Auto-write not allowed:
- Implied goals guessed from one request
- Sensitive medical or body-composition interpretation

Simple format suggestion:
- Goal bullets with optional “since” date when known

### `shopping.md`

Purpose:
- Stable store, brand, channel, and purchasing habits that improve practical recommendations

Should contain:
- Preferred stores or shopping channels
- Brand preferences
- Stable bulk-buying or budget habits

Should not contain:
- Specific shopping carts
- Price snapshots
- Inventory state
- Food database entries

Auto-write allowed:
- Explicit, repeated shopping habits
- Explicit store or brand preferences

Auto-write not allowed:
- One-time purchases
- Assumed preferences based on a single recommendation

Simple format suggestion:
- Store/channel bullets and brand/habit bullets

## Auto-Write Rules

First version should stay conservative.

Auto-write is appropriate only for:
- Explicit allergies or intolerances
- Explicit dietary goals
- Explicit and stable food preferences
- Explicit and stable shopping habits

Auto-write is not appropriate for:
- Inferred health judgments
- Sensitive identity information
- One-time comments
- Temporary plans or moods
- Content the agent is only guessing from context

When in doubt, the agent should avoid writing.

## What Is Not Long-Term Memory

The following do **not** belong in `/memories/users/<user_id>/*.md` as concrete records:

- Meal logs
- Weight history
- Apple Health sync state
- Future packaged-food database or food catalog

These are domain records, not long-term markdown memory.

Long-term memory may contain only a high-level stable summary derived from such records, for example:
- “User usually prefers high-protein dinners”

It should not store detailed event history.

## Persistence Direction

Future persistence should preserve the current contract:

- Agent still sees `/memories/users/<user_id>/*.md`
- `StoreBackend` remains the long-term memory route
- The system swaps `InMemoryStore()` for a durable store adapter

The persistence change should happen **below** the `/memories/...` interface, not by creating a second memory subsystem and not by returning to the old `app/memory/` retrieval design.

## Implementation Guidance For Later

When persistence work starts:

- Keep Phase 1 focused on `profile.md` and `preferences.md`
- Add `goals.md` and `shopping.md` only after the write rules are stable
- Treat markdown files as durable user memory summaries, not as append-only event logs
