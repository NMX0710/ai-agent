import getpass
import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

'''
Currently use tavily search api, 1000 free search per day
'''
# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

web_search = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=False,
    include_raw_content=False,
    include_images=False,
    search_depth="basic"
)