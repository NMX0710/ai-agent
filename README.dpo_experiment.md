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


## Current Status

As of commit `99422c6`, this branch has completed the first offline export stage for English `prompt + chosen` data.

Completed work:
- The branch remains based on the pre-planner LangGraph + RAG + tools path selected from commit `71afd37`.
- The runtime and prompt path were converted to English for this experiment branch.
- The batch runner was hardened for direct execution, `.env` loading, incremental JSONL writes, overwrite control, and failure logging.
- A dedicated English scenario set was added at `data/dpo_batch_scenarios_500.json`.
- The generated chosen dataset was produced successfully and stored at `data/dpo_prompt_chosen_500.jsonl`.
- The 500-sample batch finished with `500` successes and `0` failures.

Environment notes:
- This branch was run successfully in an isolated virtual environment at `/venv/diet-agent-dpo`.
- `/venv/main` was intentionally left untouched for PyTorch and other existing workloads.
- `.env` is expected locally for secrets and is ignored by git.

Primary files to read first in a future session:
- `README.dpo_experiment.md`
- `scripts/run_dpo_batch.py`
- `app/recipe_app.py`
- `data/dpo_batch_scenarios_500.json`
- `data/dpo_prompt_chosen_500.jsonl`

Next expected step:
- Generate `rejected` responses for the same English prompts with the target base model.
- After that, construct preference pairs for LoRA + DPO training.

Not done yet:
- No rejected generation has been committed in this branch yet.
- No DPO training has been started in this branch yet.
