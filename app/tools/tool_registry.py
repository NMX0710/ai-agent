# app/tools/tool_registry.py
import os
import asyncio
from typing import List, Tuple

from app.tools.file_operation_tool import read_file, write_file
from app.tools.pdf_generation_tool import generate_pdf
from app.tools.web_search_tool import web_search
from app.tools.web_scraping_tool import scrape_web_page
from app.tools.resource_download_tool import download_resource
from app.tools.terminal_operation_tool import execute_terminal_command

# ✅ 基础（本地）工具列表：和你现在的一样
BASE_TOOLS = [
    read_file,
    write_file,
    generate_pdf,
    web_search,
    scrape_web_page,
    download_resource,
    execute_terminal_command,
]

# ✅ 动态加载 MCP 工具并合并
def load_all_tools_with_mcp() -> Tuple[List, object | None]:
    """
    返回：(合并后的工具列表, mcp_handle)
    - 如果未配置 SPOONACULAR_API_KEY 或 MCP 启动失败，就只返回 BASE_TOOLS，handle=None
    """
    tools = list(BASE_TOOLS)
    mcp_handle = None

    # 延迟导入，避免没有 mcp 依赖时影响本地工具
    try:
        from app.tools.mcp_client_tools import build_mcp_client_tools
    except Exception:
        # 没装 mcp / fastmcp 也能正常跑
        return tools, None

    # 没有 key 就不启 MCP（你也可以改成抛错）
    if not os.getenv("SPOONACULAR_API_KEY"):
        return tools, None

    try:
        handle, mcp_tools = asyncio.run(
            build_mcp_client_tools(
                server_cmd="python",
                server_args=["app/mcp_servers/nutrition_mcp_server.py"],
                env={"SPOONACULAR_API_KEY": os.getenv("SPOONACULAR_API_KEY", "")},
            )
        )
        tools.extend(mcp_tools)
        mcp_handle = handle
    except Exception:
        # 启动失败也不影响本地工具
        mcp_handle = None

    return tools, mcp_handle
