import asyncio
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.observability import configure_logging
from app.recipe_app import RecipeApp


CASES = [
    {"name": "curry_rice", "prompt": "咖喱饭的热量是多少？我放了胡萝卜和鸡肉"},
    {"name": "fried_rice", "prompt": "炒饭的热量大概是多少？我放了鸡蛋和虾仁"},
    {"name": "pasta_meat_sauce", "prompt": "意大利面配牛肉酱的热量大概是多少？"},
    {"name": "chicken_sandwich", "prompt": "鸡胸肉、生菜和酱的三明治热量大概是多少？"},
    {"name": "trader_joes_mandarin_orange_chicken", "prompt": "Trader Joe's Mandarin Orange Chicken 的热量是多少？"},
    {"name": "quest_bar", "prompt": "Quest Chocolate Chip Cookie Dough protein bar 的热量是多少？"},
    {"name": "subway_turkey_6_inch", "prompt": "Subway 6-inch turkey sandwich 的热量是多少？"},
    {"name": "big_mac", "prompt": "一个巨无霸 Big Mac 的热量大概是多少？"},
    {"name": "banana", "prompt": "一根中等大小香蕉的热量是多少？"},
    {"name": "california_roll", "prompt": "一份加州卷寿司的热量大概是多少？"},
]


def _selected_cases(case_name: str | None) -> list[dict[str, str]]:
    if not case_name:
        return CASES
    selected = [case for case in CASES if case["name"] == case_name]
    if not selected:
        valid = ", ".join(case["name"] for case in CASES)
        raise SystemExit(f"Unknown case {case_name!r}. Valid cases: {valid}")
    return selected


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", dest="case_name")
    parser.add_argument("--base-url", dest="base_url")
    args = parser.parse_args()

    chat_prefix = f"nutrition-regression-{uuid4().hex[:8]}"
    user_id = "local-regression"
    results: list[dict[str, str]] = []
    selected_cases = _selected_cases(args.case_name)

    if args.base_url:
        base_url = args.base_url.rstrip("/")
        for case in selected_cases:
            chat_id = f"{chat_prefix}-{case['name']}"
            print(f"RUNNING {case['name']}: {case['prompt']}", flush=True)
            payload = json.dumps(
                {"chat_id": chat_id, "user_id": user_id, "message": case["prompt"]},
                ensure_ascii=False,
            ).encode("utf-8")
            req = Request(
                f"{base_url}/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except HTTPError as exc:
                raise SystemExit(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')}") from exc
            except URLError as exc:
                raise SystemExit(f"Request failed: {exc}") from exc
            row = {"case": case["name"], "prompt": case["prompt"], "response": body.get("response", "")}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return

    app = RecipeApp()
    try:
        for case in selected_cases:
            chat_id = f"{chat_prefix}-{case['name']}"
            print(f"RUNNING {case['name']}: {case['prompt']}", flush=True)
            response = await app.chat(chat_id=chat_id, message=case["prompt"], user_id=user_id)
            row = {"case": case["name"], "prompt": case["prompt"], "response": response}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        await app.close()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
