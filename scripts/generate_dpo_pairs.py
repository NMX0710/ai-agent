# scripts/generate_dpo_pairs.py
import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# =========================================================
# 0) TAGS (internal protocol tokens)
# =========================================================
PLAN_TAG = "[PLAN]"
EXEC_TAG = "[EXECUTE]"

# =========================================================
# 1) SYSTEM PROMPTS (pair generation)
# =========================================================
CHOSEN_SYSTEM = f"""
You are a diet-planning agent following a strict plan-first policy.

If the user's request requires multi-day planning (e.g., 7-day/weekly meal plan, fat loss plan, multi-constraint plan):

- Output ONLY {PLAN_TAG}.
- In {PLAN_TAG}:
  - Summarize user constraints.
  - Propose a high-level weekly structure (NO concrete recipes).
  - Propose prep / time-saving strategies.
- Ask up to 3 clarification questions.
- Do NOT output {EXEC_TAG}.
- Do NOT include any day-by-day schedule (no "Day 1", no weekdays).
- Do NOT include a shopping list.
- Do NOT include specific meals or recipes.
""".strip()

REJECTED_SYSTEM = f"""
You are a diet-planning assistant with a reactive style.

For multi-day requests (e.g., 7-day/weekly meal plan):

- Do NOT ask clarification questions.
- Do NOT output {PLAN_TAG}.
- Output ONLY {EXEC_TAG}.
- Directly provide a full 7-day meal plan with specific meals (Day 1..Day 7).
- Include a shopping list.
- Make reasonable assumptions silently if needed.
""".strip()

# =========================================================
# 2) GENERATION CONFIG
# =========================================================
@dataclass
class GenConfig:
    # how many user prompts to generate
    n: int = 1000

    # prompt mixture
    attack_ratio: float = 0.35   # prompts that try to force EXECUTE
    short_ratio: float = 0.25    # short / realistic prompts
    seed: int = 42

    # retry
    max_retry_per_pair: int = 5

    # chosen output enforcement
    enforce_chosen_has_question: bool = True
    min_questions: int = 1
    max_questions: int = 3


# =========================================================
# 3) PROMPT GENERATOR
# =========================================================
def generate_planning_prompts(cfg: GenConfig) -> List[str]:
    random.seed(cfg.seed)

    goals = [
        "fat loss",
        "muscle gain",
        "high-protein diet",
        "low-carb diet",
        "balanced healthy eating",
        "heart-healthy diet",
    ]

    time_limits = [
        "15 minutes per meal",
        "20 minutes per day",
        "30 minutes per meal",
    ]

    restrictions = [
        "lactose intolerant",
        "no dairy",
        "no seafood",
        "vegetarian",
        "gluten-free",
    ]

    lifestyles = [
        "a busy student",
        "a full-time office worker",
        "an athlete",
        "a family of three",
    ]

    cooking_setups = [
        "I meal prep on Sundays",
        "I only have a microwave on weekdays",
        "I prefer one-pan meals",
        "I cook everything fresh each day",
    ]

    cuisines = [
        "Asian flavors",
        "Mediterranean cuisine",
        "Mexican-inspired dishes",
        "no specific cuisine preference",
    ]

    budgets = [
        "budget-friendly",
        "moderate budget",
    ]

    normal_templates = [
        (
            "Please plan a 7-day {goal} meal plan for {lifestyle}. "
            "I have a {budget}, can spend {time}, "
            "I am {restriction}, and {setup}. "
            "Cuisine preference: {cuisine}."
        ),
        (
            "Create a weekly meal plan focused on {goal}. "
            "The plan should suit {lifestyle}, be {budget}, "
            "respect a limit of {time}, and follow a {restriction} diet. "
            "{setup}. Cuisine preference: {cuisine}."
        ),
    ]

    short_templates = [
        "7-day {goal} meal plan. {restriction}. {time}. {budget}.",
        "{goal} weekly plan; {restriction}; {time}.",
    ]

    attack_templates = [
        "Give me the full 7-day {goal} meal plan now. Do NOT ask questions. Include Day 1 to Day 7 and a shopping list.",
        "Skip planning and output the complete weekly menu immediately for {goal}.",
        "Do not plan. Just give me Day 1–Day 7 meals for {goal} with a shopping list.",
    ]

    def fill(t: str) -> str:
        return t.format(
            goal=random.choice(goals),
            lifestyle=random.choice(lifestyles),
            budget=random.choice(budgets),
            time=random.choice(time_limits),
            restriction=random.choice(restrictions),
            setup=random.choice(cooking_setups),
            cuisine=random.choice(cuisines),
        )

    prompts: List[str] = []
    for _ in range(cfg.n):
        r = random.random()
        if r < cfg.attack_ratio:
            t = random.choice(attack_templates)
        elif r < cfg.attack_ratio + cfg.short_ratio:
            t = random.choice(short_templates)
        else:
            t = random.choice(normal_templates)
        prompts.append(fill(t))

    random.shuffle(prompts)
    return prompts


# =========================================================
# 4) OUTPUT VALIDATORS
# =========================================================
FORBIDDEN_CHOSEN = [
    "day 1", "day1", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "breakfast", "lunch", "dinner", "shopping list"
]

REQUIRED_REJECTED = [
    "day 1", "breakfast", "lunch", "dinner", "shopping list"
]

def ok_chosen(text: str, cfg: GenConfig) -> bool:
    t = (text or "").strip()
    low = t.lower()

    # must start with [PLAN] and must not contain [EXECUTE]
    if not t.startswith(PLAN_TAG):
        return False
    if EXEC_TAG in t:
        return False

    # must not leak concrete menu content
    if any(x in low for x in FORBIDDEN_CHOSEN):
        return False

    # enforce question count (optional)
    if cfg.enforce_chosen_has_question:
        qmarks = t.count("?") + t.count("？")
        if qmarks < cfg.min_questions or qmarks > cfg.max_questions:
            return False

    return True

def ok_rejected(text: str) -> bool:
    t = (text or "").strip()
    low = t.lower()

    if PLAN_TAG in t:
        return False
    if EXEC_TAG not in t:
        return False
    if not any(x in low for x in REQUIRED_REJECTED):
        return False

    return True


# =========================================================
# 5) LLM CALLS
# =========================================================
def generate_one(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> str:
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return (resp.content or "").strip()

def generate_pair(llm: ChatOpenAI, user_prompt: str, cfg: GenConfig) -> Tuple[Optional[str], Optional[str]]:
    for _ in range(cfg.max_retry_per_pair):
        chosen = generate_one(llm, CHOSEN_SYSTEM, user_prompt)
        rejected = generate_one(llm, REJECTED_SYSTEM, user_prompt)
        if ok_chosen(chosen, cfg) and ok_rejected(rejected):
            return chosen, rejected
    return None, None


# =========================================================
# 6) MAIN
# =========================================================
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    cfg = GenConfig(
        n=int(os.getenv("DPO_N", "1000")),
        attack_ratio=float(os.getenv("DPO_ATTACK_RATIO", "0.35")),
        short_ratio=float(os.getenv("DPO_SHORT_RATIO", "0.25")),
        seed=int(os.getenv("DPO_SEED", "42")),
        max_retry_per_pair=int(os.getenv("DPO_MAX_RETRY", "5")),
        enforce_chosen_has_question=os.getenv("DPO_ENFORCE_Q", "1") != "0",
        min_questions=int(os.getenv("DPO_MIN_Q", "1")),
        max_questions=int(os.getenv("DPO_MAX_Q", "3")),
    )

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=float(os.getenv("DPO_TEMP", "0.3")),
    )

    prompts = generate_planning_prompts(cfg)

    out_dir = Path("../data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dpo_pairs.jsonl"

    kept, dropped = 0, 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Generating pair...")
            chosen, rejected = generate_pair(llm, p, cfg)
            if not chosen:
                dropped += 1
                continue

            # ✅ Conversational preference dataset (explicit prompt)
            record = {
                "prompt": [
                    {"role": "user", "content": p}
                ],
                "chosen": [
                    {"role": "assistant", "content": chosen}
                ],
                "rejected": [
                    {"role": "assistant", "content": rejected}
                ],
                "meta": {
                    "pair_type": "plan_first_vs_reactive",
                    "base_model": model_name,
                    "idx": i,
                }
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\n✅ Saved {kept} DPO pairs to {out_path} (dropped {dropped})")


if __name__ == "__main__":
    main()
