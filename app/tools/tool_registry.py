# app/tools/tool_registry.py
import os
import asyncio
import logging
from typing import List, Tuple

from app.tools.file_operation_tool import read_file, write_file
from app.tools.pdf_generation_tool import generate_pdf
from app.tools.web_search_tool import web_search
from app.tools.web_scraping_tool import scrape_web_page
from app.tools.resource_download_tool import download_resource
from app.tools.terminal_operation_tool import execute_terminal_command

BASE_TOOLS = [
    read_file,
    write_file,
    generate_pdf,
    web_search,
    scrape_web_page,
    download_resource,
    execute_terminal_command,
]

logging.info("[ToolRegistry] BASE_TOOLS loaded:")
for t in BASE_TOOLS:
    logging.info(f"   ├── {getattr(t, 'name', type(t).__name__)}")

async def load_all_tools_with_mcp_async() -> Tuple[List, object | None]:
    tools = list(BASE_TOOLS)
    mcp_handle = None
    try:
        from app.tools.mcp_client_tools import build_mcp_client_tools
    except Exception as e:
        logging.warning(f"[ToolRegistry] MCP client not available: {e}")
        return tools, None

    if not os.getenv("SPOONACULAR_API_KEY"):
        logging.info("[ToolRegistry] No SPOONACULAR_API_KEY → MCP disabled.")
        return tools, None

    try:
        handle, mcp_tools = await build_mcp_client_tools(
            server_cmd="python",
            server_args=["app/mcp_servers/nutrition_mcp_server.py"],
            env={"SPOONACULAR_API_KEY": os.getenv("SPOONACULAR_API_KEY", "")},
        )
        tools.extend(mcp_tools)
        mcp_handle = handle
        logging.info(f"[ToolRegistry] MCP tools loaded ({len(mcp_tools)}):")
        for t in mcp_tools:
            logging.info(f"   ├── {getattr(t, 'name', type(t).__name__)}")
    except Exception as e:
        logging.exception(f"[ToolRegistry] MCP startup failed: {e}")
        mcp_handle = None

    logging.info(f"[ToolRegistry] Total tools ready: {len(tools)}")
    return tools, mcp_handle

def load_all_tools_with_mcp() -> Tuple[List, object | None]:
    """
    同步入口：如果检测到已有事件循环（例如 uvicorn 启动时），
    为了不报错，直接返回 BASE_TOOLS，并打印提示。
    在脚本/无事件循环环境下才用 asyncio.run 跑异步加载。
    """
    # 提供一个开关，想彻底关 MCP 时可 export DISABLE_MCP=1
    if os.getenv("DISABLE_MCP") == "1":
        logging.info("[ToolRegistry] MCP disabled by DISABLE_MCP=1")
        return list(BASE_TOOLS), None

    try:
        asyncio.get_running_loop()
        logging.info("[ToolRegistry] Detected running event loop → use BASE_TOOLS only for now.")
        return list(BASE_TOOLS), None
    except RuntimeError:
        # 没有事件循环，安全使用 asyncio.run
        return asyncio.run(load_all_tools_with_mcp_async())
