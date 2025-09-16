import os
import sys
import json
import pytest

pytestmark = pytest.mark.integration  # 显式标记为“集成测试”

RECIPE_QUERY = "high protein chicken salad"


def _require_env(var: str):
    val = os.getenv(var)
    if not val:
        pytest.fail(f"Missing required environment variable: {var}")
    return val


@pytest.mark.asyncio
async def test_mcp_server_tool_invocation_real():
    """
    真实起 nutrition MCP（stdio），真实调用 search_recipe。
    断言返回 results 非空且包含关键字段。
    """
    # 确保有 key（不 mock）
    _require_env("SPOONACULAR_API_KEY")

    # 延迟导入（避免没安装 mcp 时提前失败）
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,  # 比 "python" 更稳
        args=["app/mcp_servers/nutrition_mcp_server.py"],
        env={"SPOONACULAR_API_KEY": os.environ["SPOONACULAR_API_KEY"]},
    )

    client_cm = stdio_client(params)
    read, write = await client_cm.__aenter__()
    session_cm = ClientSession(read, write)
    session = await session_cm.__aenter__()
    try:
        await session.initialize()

        listed = await session.list_tools()
        names = [t.name for t in listed.tools]
        assert "search_recipe" in names, f"tools listed: {names}"

        result = await session.call_tool("search_recipe", {"q": RECIPE_QUERY, "number": 2})
        data = getattr(result, "structuredContent", None)

        if data is None:
            # 兼容 content/text 形式
            texts = [getattr(c, "text", "") for c in getattr(result, "content", []) if getattr(c, "text", None)]
            joined = "\n".join(texts)
            try:
                data = json.loads(joined)
            except Exception:
                data = {}

        assert isinstance(data, dict) and "results" in data, f"unexpected payload: {data}"
        assert len(data["results"]) > 0
        first = data["results"][0]
        for key in ("title", "calories"):
            assert key in first
    finally:
        await session_cm.__aexit__(None, None, None)
        await client_cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_recipe_app_end_to_end_real():
    """
    使用你写的 RecipeApp 做一次完整 chat（无任何 mock）。
    需要真实 DASHSCOPE_API_KEY（embeddings）+ SPOONACULAR_API_KEY（MCP）。
    """
    _require_env("DASHSCOPE_API_KEY")
    _require_env("SPOONACULAR_API_KEY")

    # 重要：确保 PYTHONWARNINGS 不阻碍日志；可选
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    # 延迟导入，使用你真实的实现
    from app.recipe_app import RecipeApp, HumanMessage

    app = RecipeApp()

    # 直接走你的对话入口（内部会调用 graph -> retrieve -> model）
    reply = await app.chat("it-real-1", "推荐一个高蛋白鸡肉沙拉")  # thread_id 已在 chat 内使用
    assert isinstance(reply, str) and len(reply) > 0

    # （可选）再次问一个问题，看看记忆/上下文是否工作
    reply2 = await app.chat("it-real-1", "再给我一个不同口味的鸡肉沙拉")
    assert isinstance(reply2, str) and len(reply2) > 0

    # （可选）优雅关闭 MCP
    if hasattr(app, "close"):
        try:
            await app.close()
        except Exception:
            pass
