import os
import sys
import json
import pytest

# Explicitly mark these tests as integration tests
pytestmark = pytest.mark.integration

RECIPE_QUERY = "high protein chicken salad"


def _require_env(var: str):
    """
    Ensure a required environment variable is present.
    Fail the test immediately if it is missing.
    """
    val = os.getenv(var)
    if not val:
        pytest.fail(f"Missing required environment variable: {var}")
    return val


@pytest.mark.asyncio
async def test_mcp_server_tool_invocation_real():
    """
    Start the nutrition MCP server over stdio and invoke `search_recipe` for real.
    Assert that:
      - the tool is listed
      - the response contains non-empty results
      - key nutrition fields are present
    """
    # Ensure API key is available (no mocking)
    _require_env("SPOONACULAR_API_KEY")

    # Lazy imports to avoid failing early if MCP is not installed
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,  # More reliable than plain "python"
        args=["app/mcp_servers/nutrition_mcp_server.py"],
        env={"SPOONACULAR_API_KEY": os.environ["SPOONACULAR_API_KEY"]},
    )

    client_cm = stdio_client(params)
    read, write = await client_cm.__aenter__()
    session_cm = ClientSession(read, write)
    session = await session_cm.__aenter__()

    try:
        # Important: initialize session to negotiate capabilities
        await session.initialize()

        listed = await session.list_tools()
        names = [t.name for t in listed.tools]
        assert "search_recipe" in names, f"tools listed: {names}"

        result = await session.call_tool(
            "search_recipe",
            {"q": RECIPE_QUERY, "number": 2},
        )

        data = getattr(result, "structuredContent", None)

        if data is None:
            # Fallback for content/text-based responses
            texts = [
                getattr(c, "text", "")
                for c in getattr(result, "content", [])
                if getattr(c, "text", None)
            ]
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
        # Ensure graceful shutdown
        await session_cm.__aexit__(None, None, None)
        await client_cm.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_recipe_app_end_to_end_real():
    """
    Run a full end-to-end chat using the real RecipeApp implementation (no mocks).

    Requirements:
      - DASHSCOPE_API_KEY (for embeddings)
      - SPOONACULAR_API_KEY (for MCP nutrition tool)
    """
    _require_env("DASHSCOPE_API_KEY")
    _require_env("SPOONACULAR_API_KEY")

    # Optional: suppress noisy warnings during integration tests
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    # Lazy import to ensure we use the real implementation
    from app.recipe_app import RecipeApp, HumanMessage

    app = RecipeApp()

    # Run a real chat turn (internally: graph -> retrieve -> model)
    reply = await app.chat(
        "it-real-1",
        "Recommend a high-protein chicken salad",
    )
    assert isinstance(reply, str) and len(reply) > 0

    # Optional: follow-up question to test memory / conversation context
    reply2 = await app.chat(
        "it-real-1",
        "Give me another chicken salad with a different flavor profile",
    )
    assert isinstance(reply2, str) and len(reply2) > 0

    # Optional: gracefully close MCP resources if supported
    if hasattr(app, "close"):
        try:
            await app.close()
        except Exception:
            pass
