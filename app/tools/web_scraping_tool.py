import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool(description="Scrape the content of a web page")
def scrape_web_page(url: str) -> str:
    """
    Fetch and parse the HTML content of a web page.

    Args:
        url: The URL of the web page to scrape

    Returns:
        The full HTML content of the page, or an error message
    """
    try:
        # Set request headers to mimic a real browser
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        # Send HTTP GET request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for non-200 responses

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Return the full HTML document (equivalent to document.html())
        return str(soup)

    except requests.exceptions.RequestException as e:
        return f"Error scraping web page: {str(e)}"
    except Exception as e:
        return f"Error scraping web page: {str(e)}"
