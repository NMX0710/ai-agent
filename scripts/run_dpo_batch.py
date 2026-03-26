import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



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
        key = key.strip()
        value = value.strip().strip("'\"")
        if not os.environ.get(key):
            os.environ[key] = value


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


def load_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("sample_id")
            if sample_id is not None:
                sample_ids.add(str(sample_id))
    return sample_ids


def maybe_reset_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    if not overwrite:
        return

    for path in paths:
        if path.exists():
            path.unlink()


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output and failure files before starting.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry sample_ids already present in the failure log.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    from app.recipe_app import RecipeApp

    scenarios = load_scenarios(args.input)

    output_path = Path(args.output)
    failure_path = Path(args.failures)
    maybe_reset_outputs((output_path, failure_path), overwrite=args.overwrite)

    completed_ids = load_existing_sample_ids(output_path)
    failed_ids = load_existing_sample_ids(failure_path)
    skipped_existing = 0
    retried_failures = 0

    app = RecipeApp()
    success_count = 0
    failure_count = 0

    for index, scenario in enumerate(scenarios, start=1):
        sample_id = str(scenario.get("sample_id", index))
        user_input = scenario["user_input"]

        if sample_id in completed_ids:
            skipped_existing += 1
            if index % max(args.progress_every, 1) == 0:
                print(
                    f"[progress] processed={index}/{len(scenarios)} "
                    f"succeeded={success_count} failed={failure_count} "
                    f"skipped_existing={skipped_existing} retried_failures={retried_failures}"
                )
            continue

        if sample_id in failed_ids and not args.retry_failures:
            skipped_existing += 1
            if index % max(args.progress_every, 1) == 0:
                print(
                    f"[progress] processed={index}/{len(scenarios)} "
                    f"succeeded={success_count} failed={failure_count} "
                    f"skipped_existing={skipped_existing} retried_failures={retried_failures}"
                )
            continue

        if sample_id in failed_ids and args.retry_failures:
            retried_failures += 1

        try:
            record = app.generate_dpo_record(
                sample_id=sample_id,
                user_input=user_input,
                conversation_history=scenario.get("conversation_history"),
                metadata=scenario.get("metadata"),
                chat_id=scenario.get("chat_id"),
            )
            append_jsonl(output_path, record)
            completed_ids.add(sample_id)
            success_count += 1
        except Exception as exc:
            append_jsonl(
                failure_path,
                {
                    "sample_id": sample_id,
                    "user_input": user_input,
                    "metadata": scenario.get("metadata", {}),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            failed_ids.add(sample_id)
            failure_count += 1

        if index % max(args.progress_every, 1) == 0:
            print(
                f"[progress] processed={index}/{len(scenarios)} "
                f"succeeded={success_count} failed={failure_count} "
                f"skipped_existing={skipped_existing} retried_failures={retried_failures}"
            )

    print(
        f"[done] succeeded={success_count} failed={failure_count} "
        f"skipped_existing={skipped_existing} retried_failures={retried_failures} "
        f"output={output_path} failures={failure_path}"
    )


if __name__ == "__main__":
    main()
