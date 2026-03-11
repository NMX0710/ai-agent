# TODO

## Apple Health Bridge
- Implement real iOS companion bridge worker to poll `/integrations/apple-health/pending-writes`.
- Report write outcomes back to `/integrations/apple-health/write-result`.
- Add Telegram final notification after bridge success/failure callback.

## Nutrition Estimation Quality
- Add bilingual query expansion (Chinese dish -> English equivalent) before USDA/Spoonacular calls.
- Add Tavily nutrition fallback tool after USDA/Spoonacular misses and before LLM fallback.
- Attach source label in user-facing estimate card (`USDA`, `Spoonacular`, `Tavily`, `Estimated`).

## Tool Contracts
- Add explicit `food_query` field to `prepare_meal_log` contract and reject wrapper sentences.
- Keep `meal_description` for user-facing copy; use `food_query` for lookup only.

## Product UX
- Localize callback confirmation/cancel post-click messages to match user language.
- Add low-confidence warning in confirmation card when fallback source is `llm_fallback` or `hard_fallback`.

## Observability
- Add trace IDs into meal estimate logs for cross-request correlation.
- Add counters for source selection distribution (`usda/spoonacular/tavily/llm/hard`).
