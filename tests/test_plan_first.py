# tests/test_plan_first_print.py
import pytest
from app.recipe_app import RecipeApp


@pytest.mark.asyncio
async def test_plan_first_behavior_print():
    """
    This test is intentionally lightweight.
    It prints the user inputs and the raw model outputs
    so that we can manually inspect agent behavior.
    """

    app = RecipeApp()
    chat_id = "plan-first-debug-session"

    # -------------------------
    # Turn 1: plan-first request
    # -------------------------
    prompt_1 = (
        "Please help me plan a 7-day fat-loss meal plan. "
        "I can spend no more than 20 minutes cooking per day, "
        "I am lactose intolerant, and my budget is moderate."
    )

    print("\n================ TURN 1 =================")
    print("USER:")
    print(prompt_1)

    resp_1 = await app.chat(chat_id=chat_id, message=prompt_1)

    print("\nAGENT RESPONSE:")
    print(resp_1)

    # Optional minimal sanity check (can be removed if you want zero assertions)
    # assert "[PLAN]" in resp_1, "Expected PLAN section in plan-first response"

    # -------------------------
    # Turn 2: direct QA request
    # -------------------------
    prompt_2 = "How many calories are in chicken breast?"

    print("\n================ TURN 2 =================")
    print("USER:")
    print(prompt_2)

    resp_2 = await app.chat(chat_id=chat_id, message=prompt_2)

    print("\nAGENT RESPONSE:")
    print(resp_2)

    # Optional minimal sanity check
    # assert "[EXECUTE]" in resp_2, "Expected EXECUTE section in direct QA response"
