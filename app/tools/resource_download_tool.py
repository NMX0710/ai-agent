import requests
import os
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.tools import tool
from ..settings import FILE_SAVE_DIR


@tool(description="Download a resource from a given URL")
def download_resource(url: str, file_name: str) -> str:
    """
    Download a resource from the given URL and save it locally.

    Args:
        url: The URL of the resource to download
        file_name: The filename to use when saving the resource

    Returns:
        A success message or an error message
    """
    try:
        # Ensure the download directory exists
        download_dir = FILE_SAVE_DIR
        download_dir.mkdir(parents=True, exist_ok=True)

        # Construct the full file path
        file_path = download_dir / file_name

        # Set request headers to mimic a browser
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        # Send HTTP request and stream the response
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Write the downloaded content to disk
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)

        return f"Resource downloaded successfully to: {file_path}"

    except requests.exceptions.RequestException as e:
        return f"Error downloading resource: {str(e)}"
    except IOError as e:
        return f"Error writing file: {str(e)}"
    except Exception as e:
        return f"Error downloading resource: {str(e)}"
