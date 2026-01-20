# app/tools/mcp_client_tools.py
import os
import asyncio
import json
from typing import Type, List, Tuple, Optional, Dict

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client  # Official example usage


class McpToolInput(BaseModel):
    """Default input schema for MCP tools."""
    payload: dict = Field(default_factory=dict)


class McpTool(BaseTool):
    """
    A LangChain Tool wrapper around an MCP tool.

    This class delegates tool execution to an MCP ClientSession via call_tool().
    """
    name: str
    description: str
    tool_name_on_mcp: str
    session: ClientSession
    args_schema: Type[BaseModel] = McpToolInput

    def _run(self, payload: dict) -> str:
        # BaseTool sync entrypoint; delegate to async implementation.
        return asyncio.run(self._arun(payload))

    async def _arun(self, payload: dict) -> str:
        result = await self.session.call_tool(self.tool_name_on_mcp, payload)

        # Always return a JSON string so the LLM receives a consistent format.
        return json.dumps({
            "structured": getattr(result, "structuredContent", None),
            "content": [getattr(c, "text", None) for c in getattr(result, "content", [])],
        })


class McpClientHandle:
    """
    Holds async context managers so the MCP session can be closed gracefully.
    Useful for FastAPI shutdown hooks.
    """
    def __init__(self, client_cm, session_cm, session: ClientSession):
        self.client_cm = client_cm
        self.session_cm = session_cm
        self.session = session

    async def close(self):
        # Close session first, then close the underlying stdio client.
        await self.session_cm.__aexit__(None, None, None)
        await self.client_cm.__aexit__(None, None, None)


async def build_mcp_client_tools(
    server_cmd: str = "python",
    server_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[McpClientHandle, List[McpTool]]:
    """
    Start an MCP server over stdio and wrap its exposed tools as LangChain Tools.

    Defaults to launching the local nutrition MCP server:
      - command: python
      - args: ["app/mcp_servers/nutrition_mcp_server.py"]

    Returns:
        (handle, tools)
        - handle: McpClientHandle used for graceful shutdown
        - tools: List of LangChain tools backed by MCP
    """
    server_args = server_args or ["app/mcp_servers/nutrition_mcp_server.py"]
    env = env or {}

    # 1) Configure stdio server parameters (recommended approach in MCP docs)
    params = StdioServerParameters(command=server_cmd, args=server_args, env=env)

    # 2) Open the stdio client and create a persistent session
    client_cm = stdio_client(params)
    read, write = await client_cm.__aenter__()

    session_cm = ClientSession(read, write)
    session: ClientSession = await session_cm.__aenter__()

    # Important: initialize the session to negotiate capabilities
    await session.initialize()

    # 3) List server tools and wrap them into LangChain tools
    resp = await session.list_tools()
    tools: List[McpTool] = []

    for t in resp.tools:
        tools.append(
            McpTool(
                name=f"mcp_{t.name}",
                description=t.description or "MCP tool",
                tool_name_on_mcp=t.name,
                session=session,
            )
        )

    handle = McpClientHandle(client_cm, session_cm, session)
    return handle, tools
