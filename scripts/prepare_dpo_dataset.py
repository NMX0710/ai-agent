import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare prompt/chosen/rejected data for DPO training.')
    parser.add_argument('--input', default='outputs/dpo_pairs_qwen3_8b.jsonl')
    parser.add_argument('--train-output', default='data/dpo_train_qwen3_8b.jsonl')
    parser.add_argument('--eval-output', default='data/dpo_eval_qwen3_8b.jsonl')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--eval-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-prompt-chars', type=int, default=12000)
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    formatted = []
    for record in records:
        prompt_messages = record['prompt_messages']
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(prompt) > args.max_prompt_chars:
            prompt = prompt[-args.max_prompt_chars:]

        formatted.append({
            'sample_id': record['sample_id'],
            'prompt': prompt,
            'chosen': record['chosen'].strip(),
            'rejected': record['rejected'].strip(),
            'user_input': record.get('user_input'),
            'metadata': record.get('metadata', {}),
        })

    rng = random.Random(args.seed)
    rng.shuffle(formatted)

    eval_size = min(max(args.eval_size, 1), max(len(formatted) - 1, 1))
    eval_records = formatted[:eval_size]
    train_records = formatted[eval_size:]

    write_jsonl(Path(args.train_output), train_records)
    write_jsonl(Path(args.eval_output), eval_records)

    print(f'[done] total={len(formatted)} train={len(train_records)} eval={len(eval_records)}')
    if train_records:
        print(f"[sample-train] {train_records[0]['sample_id']}")
    if eval_records:
        print(f"[sample-eval] {eval_records[0]['sample_id']}")


if __name__ == '__main__':
    main()
