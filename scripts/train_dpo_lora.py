import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


TARGET_MODULES = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]


def main() -> None:
    parser = argparse.ArgumentParser(description='Single-GPU LoRA DPO training for Qwen3-8B.')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--train-file', default='data/dpo_train_qwen3_8b.jsonl')
    parser.add_argument('--eval-file', default='data/dpo_eval_qwen3_8b.jsonl')
    parser.add_argument('--output-dir', default='outputs/dpo_lora_qwen3_8b')
    parser.add_argument('--max-length', type=int, default=1536)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--eval-steps', type=int, default=10)
    parser.add_argument('--save-steps', type=int, default=20)
    parser.add_argument('--logging-steps', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--per-device-train-batch-size', type=int, default=1)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=TARGET_MODULES,
    )

    train_dataset = load_dataset('json', data_files=args.train_file, split='train')
    eval_dataset = load_dataset('json', data_files=args.eval_file, split='train')

    training_args = DPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        report_to='none',
        remove_unused_columns=False,
        max_length=args.max_length,
        truncation_mode='keep_end',
        beta=args.beta,
        save_total_limit=2,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    metrics = train_result.metrics
    metrics['cuda_max_memory_allocated_gb'] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    metrics['cuda_max_memory_reserved_gb'] = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics['cuda_max_memory_allocated_gb'] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    eval_metrics['cuda_max_memory_reserved_gb'] = round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3)
    trainer.log_metrics('eval', eval_metrics)
    trainer.save_metrics('eval', eval_metrics)

    print('[done] training and evaluation completed')


if __name__ == '__main__':
    main()
