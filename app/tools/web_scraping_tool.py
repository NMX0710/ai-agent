import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool(description="Scrape the content of a web page")
def scrape_web_page(url: str) -> str:
    """
    抓取网页内容

    Args:
        url: 要抓取的网页URL

    Returns:
        网页HTML内容或错误信息
    """
    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # 发送GET请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 如果状态码不是200会抛出异常

        # 解析HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # 返回完整的HTML内容，对应鱼皮的document.html()
        return str(soup)

    except requests.exceptions.RequestException as e:
        return f"Error scraping web page: {str(e)}"
    except Exception as e:
        return f"Error scraping web page: {str(e)}"