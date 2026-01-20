from ..settings import FILE_SAVE_DIR
from langchain_core.tools import tool


@tool(description="Read content from a file")
def read_file(file_name: str) -> str:
    file_path = FILE_SAVE_DIR / file_name
    try:
        return file_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading {file_name}: {e}"

@tool(description="Write content to a file")
def write_file(file_name: str, content: str) -> str:
    file_path = FILE_SAVE_DIR / file_name
    try:
        FILE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return f"File {file_name} written successfully"
    except Exception as e:
        return f"Error writing {file_name}: {e}"




