# TODO

## Apple Health Bridge
- Replace mock bridge writer with a real iOS/macOS HealthKit writer.
- Build a native companion app flow for HealthKit authorization and foreground sync.
- Add Telegram final notification after bridge success/failure callback.
- Decide whether bridge sync should be manual, foreground auto-sync, or background scheduled sync.

## Nutrition Estimation Quality
- Add Tavily nutrition fallback tool after USDA/Spoonacular misses and before LLM fallback.
- Attach source label in user-facing estimate card (`USDA`, `Spoonacular`, `Tavily`, `Estimated`).
- Expand agent-side query planning so `food_query` / `food_query_en` come primarily from tool planning, not rule-based extraction.

## Tool Contracts
- Preserve the `meal_description` / `food_query` / `food_query_en` contract and audit future tools against it.
- Keep fallback normalization thin; prefer Deep Agent skill/prompt improvements over new parsing rules.

## Product UX
- Localize callback confirmation/cancel post-click messages to match user language.
- Add low-confidence warning in confirmation card when fallback source is `llm_fallback` or `hard_fallback`.
- Show source labels in the estimate preview card and post-confirmation acknowledgement.

## Observability
- Add trace IDs into meal estimate logs for cross-request correlation.
- Add counters for source selection distribution (`usda/spoonacular/tavily/llm/hard`).
- Add bridge runner metrics for claimed, synced, failed, and expired writes.
