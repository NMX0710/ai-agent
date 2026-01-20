import sys
import asyncio
from pathlib import Path

# Add project root to sys.path so "app.*" imports work when running the script directly.
project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.recipe_app import RecipeApp


async def main():
    app = RecipeApp()
    chat_id = "mem-test-runner"

    print("== Round 1 ==")
    r1 = await app.chat(chat_id, "My name is Alice. Please remember it.")
    print("A1:", r1)

    # Read the agent state from the checkpointer using the same thread_id
    config = {"configurable": {"thread_id": chat_id}}
    snap1 = app.agent_executor.get_state(config)

    # Handle both snapshot shapes (object with .values, or plain dict)
    msgs1 = (
        snap1.values.get("messages", [])
        if hasattr(snap1, "values")
        else snap1.get("messages", [])
    )
    print("Messages after round 1:", len(msgs1))

    print("\n== Round 2 ==")
    r2 = await app.chat(chat_id, "What is my name?")
    print("Q2: What is my name?")
    print("A2:", r2)

    snap2 = app.agent_executor.get_state(config)
    msgs2 = (
        snap2.values.get("messages", [])
        if hasattr(snap2, "values")
        else snap2.get("messages", [])
    )
    print("Messages after round 2:", len(msgs2))

    print("\n✅ Persisted state:", len(msgs2) > len(msgs1))


if __name__ == "__main__":
    asyncio.run(main())
