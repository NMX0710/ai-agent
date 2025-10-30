# app/tools/mcp_client_tools.py
import os, asyncio, json
from typing import Type, List, Tuple
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client  # 官方示例用法

class McpToolInput(BaseModel):
    payload: dict = Field(default_factory=dict)

class McpTool(BaseTool):
    name: str
    description: str
    tool_name_on_mcp: str
    session: ClientSession
    args_schema: Type[BaseModel] = McpToolInput

    def _run(self, payload: dict) -> str:
        return asyncio.run(self._arun(payload))

    async def _arun(self, payload: dict) -> str:
        result = await self.session.call_tool(self.tool_name_on_mcp, payload)
        # 统一转成字符串给 LLM
        return json.dumps({
            "structured": getattr(result, "structuredContent", None),
            "content": [getattr(c, "text", None) for c in getattr(result, "content", [])]
        })

class McpClientHandle:
    """保存上下文管理器，便于 FastAPI 关机时优雅关闭。"""
    def __init__(self, client_cm, session_cm, session: ClientSession):
        self.client_cm = client_cm
        self.session_cm = session_cm
        self.session = session

    async def close(self):
        await self.session_cm.__aexit__(None, None, None)
        await self.client_cm.__aexit__(None, None, None)

async def build_mcp_client_tools(
    server_cmd: str = "python",
    server_args: List[str] = None,
    env: dict = None
) -> Tuple[McpClientHandle, List[McpTool]]:
    """
    启动 nutrition MCP（stdio）并把其工具包装成 LangChain Tools
    """
    server_args = server_args or ["app/mcp_servers/nutrition_mcp_server.py"]
    env = env or {}

    # 1) 配置 stdio Server 参数（官方推荐做法）
    params = StdioServerParameters(command=server_cmd, args=server_args, env=env)

    # 2) 打开 stdio client，并创建会话（保持常驻）
    client_cm = stdio_client(params)
    read, write = await client_cm.__aenter__()
    session_cm = ClientSession(read, write)
    session: ClientSession = await session_cm.__aenter__()
    await session.initialize()  # 很关键：先初始化协商能力

    # 3) 列出工具并包装
    resp = await session.list_tools()
    tools: List[McpTool] = []
    for t in resp.tools:
        tools.append(
            McpTool(
                name=f"mcp_{t.name}",
                description=t.description or "MCP tool",
                tool_name_on_mcp=t.name,
                session=session
            )
        )

    handle = McpClientHandle(client_cm, session_cm, session)
    return handle, tools
