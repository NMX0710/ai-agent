# Offline DPO Prompt+Chosen Export

This branch is an offline data-construction environment built from commit `71afd37` plus exporter tooling. It keeps the pre-planner LangGraph + RAG + tools path and captures the closest structured model-facing input before the ReAct agent call.

## Required environment

- `OPENAI_API_KEY` for both chosen generation and RAG embeddings.
- Optional: `OPENAI_MODEL`, `OPENAI_TEMPERATURE`, and `OPENAI_EMBEDDING_MODEL`.

## Batch command

```bash
python3 scripts/run_dpo_batch.py \
  --env-file .env \
  --input data/dpo_batch_scenarios.json \
  --output outputs/dpo_prompt_chosen.jsonl \
  --failures outputs/dpo_prompt_chosen_failures.jsonl \
  --progress-every 10
```

Behavior:
- Existing success records are preserved and skipped on rerun.
- Existing failures are skipped by default; add `--retry-failures` to retry them.
- Add `--overwrite` only when you intentionally want a fresh run.
- Records are appended incrementally after each sample.

## Record fields

- `sample_id`
- `user_input`
- `retrieved_context`
- `retrieved_context_text`
- `agent_input`
- `messages_before_agent_call`
- `returned_messages`
- `chosen`
- `metadata`
- `model_provider`
- `tool_names`
