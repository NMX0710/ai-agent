import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.settings import OPENFOODFACTS_USER_AGENT
from app.tools.nutrition_http_tools import openfoodfacts_search_products


def test_openfoodfacts_search_products_sends_configured_user_agent(monkeypatch):
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"products": []}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return _FakeResponse()

    monkeypatch.setattr("app.tools.nutrition_http_tools.httpx.Client", _FakeClient)

    result = openfoodfacts_search_products.func(query="quest protein bar", page_size=3)

    assert result["count"] == 0
    assert captured["headers"]["User-Agent"] == OPENFOODFACTS_USER_AGENT
    assert captured["params"]["search_terms"] == "quest protein bar"
