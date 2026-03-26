import argparse
import json
import os
from pathlib import Path

from app.recipe_app import RecipeApp


def load_env_file(path: str | None) -> None:
    if not path:
        return

    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_scenarios(path: str):
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data

    raise ValueError("Scenario input must be a JSON array or JSONL file.")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the 71afd37 LangGraph + RAG + tools pipeline and export prompt/chosen records."
    )
    parser.add_argument(
        "--input",
        default="data/dpo_batch_scenarios.json",
        help="Path to a JSON or JSONL scenario file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/dpo_prompt_chosen.jsonl",
        help="Destination JSONL file for successful records.",
    )
    parser.add_argument(
        "--failures",
        default="outputs/dpo_prompt_chosen_failures.jsonl",
        help="Destination JSONL file for failed records.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional env file to load before constructing the app.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress after this many processed samples.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    scenarios = load_scenarios(args.input)

    output_path = Path(args.output)
    failure_path = Path(args.failures)
    if output_path.exists():
        output_path.unlink()
    if failure_path.exists():
        failure_path.unlink()

    app = RecipeApp()
    success_count = 0
    failure_count = 0

    for index, scenario in enumerate(scenarios, start=1):
        sample_id = str(scenario.get("sample_id", index))
        user_input = scenario["user_input"]
        try:
            record = app.generate_dpo_record(
                sample_id=sample_id,
                user_input=user_input,
                conversation_history=scenario.get("conversation_history"),
                metadata=scenario.get("metadata"),
                chat_id=scenario.get("chat_id"),
            )
            append_jsonl(output_path, record)
            success_count += 1
        except Exception as exc:
            append_jsonl(
                failure_path,
                {
                    "sample_id": sample_id,
                    "user_input": user_input,
                    "metadata": scenario.get("metadata", {}),
                    "error": str(exc),
                },
            )
            failure_count += 1

        if index % max(args.progress_every, 1) == 0:
            print(
                f"[progress] processed={index}/{len(scenarios)} "
                f"succeeded={success_count} failed={failure_count}"
            )

    print(
        f"[done] succeeded={success_count} failed={failure_count} "
        f"output={output_path} failures={failure_path}"
    )


if __name__ == "__main__":
    main()
