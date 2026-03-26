import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool(description="Scrape the content of a web page")
def scrape_web_page(url: str) -> str:
    """Scrape the HTML content of a web page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return str(soup)
    except requests.exceptions.RequestException as exc:
        return f"Error scraping web page: {exc}"
    except Exception as exc:
        return f"Error scraping web page: {exc}"
