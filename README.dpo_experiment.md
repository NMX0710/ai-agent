# Offline DPO Data And Training Workflow

This branch is an offline DPO data-construction and training environment built from commit `71afd37` plus exporter tooling. It keeps the pre-planner LangGraph + RAG + tools path and captures the closest structured model-facing input before the ReAct agent call.

## Required Environment

- `OPENAI_API_KEY` for both chosen generation and RAG embeddings.
- Optional: `OPENAI_MODEL`, `OPENAI_TEMPERATURE`, and `OPENAI_EMBEDDING_MODEL`.
- The validated runtime for the current workflow is `/venv/main`.

## Prompt + Chosen Export

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

## Record Fields

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

As of `2026-03-26` on branch `exp/dpo-langgraph-data-pipeline`, this repo has completed one full DPO rerun cycle after fixing truncated rejected answers.

Completed work:
- The branch remains based on the pre-planner LangGraph + RAG + tools path selected from commit `71afd37`.
- The runtime and prompt path were converted to English for this experiment branch.
- The batch runner was hardened for direct execution, `.env` loading, incremental JSONL writes, overwrite control, and failure logging.
- A dedicated English scenario set was added at `data/dpo_batch_scenarios_500.json`.
- The generated chosen dataset was produced successfully and stored at `data/dpo_prompt_chosen_500.jsonl`.
- The 500-sample batch finished with `500` successes and `0` failures.
- An initial rejected-pair file was generated at `outputs/dpo_pairs_qwen3_8b.jsonl`.
- An initial train/eval split was generated at `data/dpo_train_qwen3_8b.jsonl` and `data/dpo_eval_qwen3_8b.jsonl`.
- An initial LoRA DPO run was completed at `outputs/dpo_lora_qwen3_8b_run1`.
- A regenerated rejected-pair file was later produced at `outputs/dpo_pairs_qwen3_8b_v2.jsonl` with `max_new_tokens=1024`.
- A refreshed train/eval split was prepared from the regenerated pair file at `data/dpo_train_qwen3_8b_v2.jsonl` and `data/dpo_eval_qwen3_8b_v2.jsonl`.
- A fresh LoRA DPO rerun was completed at `outputs/dpo_lora_qwen3_8b_run2_v2`.

Environment notes:
- `/venv/main` contains the required ML stack for the DPO workflow: `torch`, `transformers`, `trl`, `peft`, `datasets`, and `accelerate`.
- CUDA is available from `/venv/main` and this machine currently exposes 2 GPUs.
- The rerun documented here used `CUDA_VISIBLE_DEVICES=0`.
- `.env` is expected locally for secrets and is ignored by git.

Primary files to read first in a future session:
- `README.dpo_experiment.md`
- `scripts/run_dpo_batch.py`
- `scripts/prepare_dpo_dataset.py`
- `scripts/train_dpo_lora.py`
- `app/recipe_app.py`
- `data/dpo_batch_scenarios_500.json`
- `data/dpo_prompt_chosen_500.jsonl`
- `outputs/dpo_pairs_qwen3_8b_v2.jsonl`
- `data/dpo_train_qwen3_8b_v2.jsonl`
- `data/dpo_eval_qwen3_8b_v2.jsonl`
- `outputs/dpo_lora_qwen3_8b_run2_v2`

## Artifact Timeline

- `outputs/dpo_pairs_qwen3_8b.jsonl`
  - Initial rejected-pair artifact.
  - Rejected generation config used `max_new_tokens=256`.
- `data/dpo_train_qwen3_8b.jsonl` and `data/dpo_eval_qwen3_8b.jsonl`
  - Prepared from the initial rejected-pair artifact.
- `outputs/dpo_lora_qwen3_8b_run1`
  - Trained from the initial train/eval split.
- `outputs/dpo_pairs_qwen3_8b_v2.jsonl`
  - Regenerated rejected-pair artifact.
  - Rejected generation config used `max_new_tokens=1024`.
- `data/dpo_train_qwen3_8b_v2.jsonl` and `data/dpo_eval_qwen3_8b_v2.jsonl`
  - Refreshed DPO dataset prepared from `outputs/dpo_pairs_qwen3_8b_v2.jsonl`.
- `outputs/dpo_lora_qwen3_8b_run2_v2`
  - Refreshed LoRA DPO run trained from the regenerated pair data.

## Important Finding

The first rejected-pair artifact was heavily truncated because the rejected-generation token cap was too small. The original `data/dpo_train_qwen3_8b.jsonl`, `data/dpo_eval_qwen3_8b.jsonl`, and `outputs/dpo_lora_qwen3_8b_run1` all came from that earlier truncated data and should not be treated as the final result for this experiment.

The refreshed v2 dataset materially improves that issue:
- `data/dpo_train_qwen3_8b_v2.jsonl`: 450 samples, no obvious rejected truncation detected in the validation pass.
- `data/dpo_eval_qwen3_8b_v2.jsonl`: 50 samples, 1 borderline sample flagged by a simple heuristic.
- Rejected response lengths increased substantially relative to the original pair file.

## Refreshed Run Metrics

Refreshed DPO run: `outputs/dpo_lora_qwen3_8b_run2_v2`

- `train_loss`: `0.04294645870686509`
- `eval_loss`: `0.00027436931850388646`
- `train_runtime`: `1064.2589` seconds
- `eval_runtime`: `15.676` seconds
- `cuda_max_memory_allocated_gb`: `21.737`
- `cuda_max_memory_reserved_gb`: `30.754`

## Training Config Snapshot

Model and method:
- Base model: `Qwen/Qwen3-8B`
- Fine-tuning method: LoRA + DPO
- Runtime: single GPU with `CUDA_VISIBLE_DEVICES=0`
- Precision: `bf16`
- Reference model: implicit (`ref_model=None` in `DPOTrainer`)

LoRA config from `scripts/train_dpo_lora.py`:
- `lora_r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`

DPO / optimization config used for the refreshed rerun:
- Train file: `data/dpo_train_qwen3_8b_v2.jsonl`
- Eval file: `data/dpo_eval_qwen3_8b_v2.jsonl`
- Train samples: `450`
- Eval samples: `50`
- `max_length=1024`
- `max_steps=180`
- `eval_steps=20`
- `save_steps=60`
- `logging_steps=5`
- `learning_rate=5e-6`
- `beta=0.1`
- `per_device_train_batch_size=1`
- `per_device_eval_batch_size=1`
- `gradient_accumulation_steps=8`
- `seed=42`
- Final trainer epoch reading: `3.16`

Dataset preparation details:
- Input pair file for the refreshed run: `outputs/dpo_pairs_qwen3_8b_v2.jsonl`
- Rejected generation cap in v2: `max_new_tokens=1024`
- Dataset prep script: `scripts/prepare_dpo_dataset.py`
- Prompt formatting used Qwen chat template with `enable_thinking=False` and `add_generation_prompt=True`
- Eval split size: `50`
- Split seed: `42`

Commands used for the refreshed rerun:

```bash
source /venv/main/bin/activate
python scripts/prepare_dpo_dataset.py \
  --input outputs/dpo_pairs_qwen3_8b_v2.jsonl \
  --train-output data/dpo_train_qwen3_8b_v2.jsonl \
  --eval-output data/dpo_eval_qwen3_8b_v2.jsonl \
  --model Qwen/Qwen3-8B \
  --eval-size 50 \
  --seed 42
```

```bash
source /venv/main/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo_lora.py \
  --train-file data/dpo_train_qwen3_8b_v2.jsonl \
  --eval-file data/dpo_eval_qwen3_8b_v2.jsonl \
  --output-dir outputs/dpo_lora_qwen3_8b_run2_v2 \
  --max-length 1024 \
  --max-steps 180 \
  --eval-steps 20 \
  --save-steps 60 \
  --logging-steps 5 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --seed 42
```

## Interview Framing

If the project is framed around steering the model toward more complete and detailed answers, the cleanest explanation is:
- We first exported `prompt + chosen` examples from the existing LangGraph + RAG pipeline.
- We generated `rejected` answers with the target base model.
- We discovered the first rejected-generation pass was truncated because the token cap was too small.
- We regenerated the rejected answers with a higher token cap (`1024`), rebuilt the DPO train/eval split, and reran LoRA DPO training.
- The rerun successfully changed model behavior relative to base Qwen3-8B on every eval sample we compared.
- If the desired product style is `more complete / more detailed / more meal-plan-like`, the rerun moved the model in that direction.
- A separate judge that favored `simple / concise / exact constraint` still preferred base more often, which shows the evaluation rubric depends on the intended product style.

## Base Vs LoRA Comparison

A refreshed qualitative comparison was run on the full `data/dpo_eval_qwen3_8b_v2.jsonl` split.

Artifacts:
- `outputs/qwen3_base_vs_lora_compare_run2_v2.jsonl`
- `outputs/qwen3_base_vs_lora_compare_run2_v2_judged.jsonl`
- `outputs/qwen3_base_vs_lora_compare_run2_v2_summary.json`

Judge summary using `gpt-4.1-mini` on 50 eval samples:
- `base`: 37 wins
- `lora`: 12 wins
- `tie`: 1

Interpretation:
- The refreshed LoRA adapter clearly changes model behavior relative to base Qwen3-8B.
- However, on this judged eval slice, the base model still wins most comparisons.
- The next iteration should focus on data quality and preference quality rather than assuming the training rerun alone is sufficient.

## Continuation Point

If work resumes in a later session, start from these assumptions:

1. The regenerated rejected-pair artifact already exists at `outputs/dpo_pairs_qwen3_8b_v2.jsonl`.
2. The refreshed dataset already exists at `data/dpo_train_qwen3_8b_v2.jsonl` and `data/dpo_eval_qwen3_8b_v2.jsonl`.
3. The refreshed LoRA run already exists at `outputs/dpo_lora_qwen3_8b_run2_v2`.
4. The judged base-vs-LoRA comparison already exists and currently favors base Qwen3-8B over the refreshed LoRA adapter on the eval slice.
