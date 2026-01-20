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

# Base tools that are always available (no MCP dependency)
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
    logging.info("   ├── %s", getattr(t, "name", type(t).__name__))


async def load_all_tools_with_mcp_async() -> Tuple[List, object | None]:
    """
    Async entrypoint: load BASE_TOOLS plus MCP tools (if available and enabled).

    Returns:
        (tools, mcp_handle)
        - tools: a list of LangChain-compatible tools
        - mcp_handle: a handle for graceful shutdown (or None if MCP is not used)
    """
    tools = list(BASE_TOOLS)
    mcp_handle = None

    # MCP client is optional; if import fails, fall back to BASE_TOOLS.
    try:
        from app.tools.mcp_client_tools import build_mcp_client_tools
    except Exception as e:
        logging.warning("[ToolRegistry] MCP client not available: %s", e)
        return tools, None

    # MCP nutrition server requires SPOONACULAR_API_KEY.
    if not os.getenv("SPOONACULAR_API_KEY"):
        logging.info("[ToolRegistry] No SPOONACULAR_API_KEY → MCP disabled.")
        return tools, None

    try:
        # Start MCP server over stdio and wrap its tools
        handle, mcp_tools = await build_mcp_client_tools(
            server_cmd="python",
            server_args=["app/mcp_servers/nutrition_mcp_server.py"],
            env={"SPOONACULAR_API_KEY": os.getenv("SPOONACULAR_API_KEY", "")},
        )

        tools.extend(mcp_tools)
        mcp_handle = handle

        logging.info("[ToolRegistry] MCP tools loaded (%d):", len(mcp_tools))
        for t in mcp_tools:
            logging.info("   ├── %s", getattr(t, "name", type(t).__name__))

    except Exception as e:
        logging.exception("[ToolRegistry] MCP startup failed: %s", e)
        mcp_handle = None

    logging.info("[ToolRegistry] Total tools ready: %d", len(tools))
    return tools, mcp_handle


def load_all_tools_with_mcp() -> Tuple[List, object | None]:
    """
    Sync entrypoint.

    If a running event loop is detected (e.g., during uvicorn startup),
    return BASE_TOOLS only to avoid asyncio.run() errors.

    In scripts / environments without an active event loop, use asyncio.run()
    to load MCP tools asynchronously.
    """
    # Optional switch to fully disable MCP: export DISABLE_MCP=1
    if os.getenv("DISABLE_MCP") == "1":
        logging.info("[ToolRegistry] MCP disabled by DISABLE_MCP=1")
        return list(BASE_TOOLS), None

    try:
        asyncio.get_running_loop()
        logging.info("[ToolRegistry] Detected running event loop → using BASE_TOOLS only for now.")
        return list(BASE_TOOLS), None
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        return asyncio.run(load_all_tools_with_mcp_async())
