import sys
from pathlib import Path

project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "tests" else Path(__file__).parent
sys.path.append(str(project_root))


TEST_MESSAGES = [
    "Search the web for a reliable lentil soup recipe.",
    "Open https://example.com and summarize the page.",
    "Run `pwd` and tell me the current working directory.",
    "Download a sample JSON file from https://jsonplaceholder.typicode.com/posts/1.",
    "Write a short shopping list to groceries.txt.",
]


def main():
    print("=== Tool smoke-test prompts ===")
    for index, message in enumerate(TEST_MESSAGES, start=1):
        print(f"{index}. {message}")
    print("=== End of tool smoke-test prompts ===")


if __name__ == "__main__":
    main()
