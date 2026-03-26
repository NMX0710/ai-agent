from langchain_core.tools import tool
import requests

from ..settings import FILE_SAVE_DIR


@tool(description="Download a resource from a given URL")
def download_resource(url: str, file_name: str) -> str:
    """Download a resource and save it in the configured file directory."""
    try:
        download_dir = FILE_SAVE_DIR
        download_dir.mkdir(parents=True, exist_ok=True)
        file_path = download_dir / file_name
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        return f"Resource downloaded successfully to: {file_path}"
    except requests.exceptions.RequestException as exc:
        return f"Error downloading resource: {exc}"
    except IOError as exc:
        return f"Error writing file: {exc}"
    except Exception as exc:
        return f"Error downloading resource: {exc}"
