import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "tests" else Path(__file__).parent
sys.path.append(str(project_root))

from app.recipe_app import RecipeApp


async def main():
    app = RecipeApp()
    chat_id = "memory-smoke-test"

    print("== Round 1 ==")
    response_one = await app.chat(chat_id, "My name is Alice. Please remember it.")
    print("A1:", response_one)

    config = {"configurable": {"thread_id": chat_id}}
    snapshot_one = app.agent_executor.get_state(config)
    messages_one = snapshot_one.values.get("messages", []) if hasattr(snapshot_one, "values") else snapshot_one.get("messages", [])
    print("Messages after round 1:", len(messages_one))

    print("\n== Round 2 ==")
    response_two = await app.chat(chat_id, "What is my name?")
    print("Q2: What is my name?")
    print("A2:", response_two)

    snapshot_two = app.agent_executor.get_state(config)
    messages_two = snapshot_two.values.get("messages", []) if hasattr(snapshot_two, "values") else snapshot_two.get("messages", [])
    print("Messages after round 2:", len(messages_two))

    print("\nPersisted:", len(messages_two) > len(messages_one))


if __name__ == "__main__":
    asyncio.run(main())
