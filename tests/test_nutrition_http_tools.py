import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app.tools.nutrition_http_tools import (
    OPENFOODFACTS_USER_AGENT,
    openfoodfacts_search_products,
    spoonacular_search_recipe,
    usda_search_foods,
)


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


def test_usda_search_foods_flags_unclear_basis_for_composed_dish_rows(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "foods": [
                    {
                        "fdcId": 1,
                        "description": "Sushi roll, California",
                        "dataType": "Survey (FNDDS)",
                        "brandName": None,
                        "servingSize": None,
                        "servingSizeUnit": None,
                        "foodNutrients": [
                            {"nutrientName": "Energy", "value": 94},
                            {"nutrientName": "Protein", "value": 2.92},
                            {"nutrientName": "Carbohydrate, by difference", "value": 18.39},
                            {"nutrientName": "Total lipid (fat)", "value": 0.67},
                        ],
                    }
                ],
                "totalHits": 1,
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, params=None, json=None):
            return _FakeResponse()

    monkeypatch.setenv("USDA_API_KEY", "test-key")
    monkeypatch.setattr("app.tools.nutrition_http_tools.httpx.Client", _FakeClient)

    result = usda_search_foods.func(query="California roll sushi", page_size=3)

    assert result["count"] == 1
    assert "Serving basis is unclear" in result["foods"][0]["basis_warning"]
    assert "query_context" not in result
    assert result["foods"][0]["is_composed_dish"] is True
    assert result["foods"][0]["serving_basis_unclear"] is True
    assert result["foods"][0]["calories_kcal"] == 94
    assert result["foods"][0]["protein_g"] == 2.92


def test_openfoodfacts_search_products_reports_provider_facts_and_basis(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "products": [
                    {
                        "code": "123",
                        "product_name": "Quest Chocolate Chip Cookie Dough Protein Bar",
                        "brands": "Quest",
                        "serving_size": "1 bar (60 g)",
                        "nutriments": {
                            "energy-kcal_serving": 200,
                            "proteins_serving": 21,
                            "carbohydrates_serving": 22,
                            "fat_serving": 8,
                        },
                    }
                ]
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            return _FakeResponse()

    monkeypatch.setattr("app.tools.nutrition_http_tools.httpx.Client", _FakeClient)

    result = openfoodfacts_search_products.func(query="Quest Chocolate Chip Cookie Dough protein bar", page_size=3)

    assert "query_context" not in result
    assert result["products"][0]["nutrition_basis"] == "serving"
    assert result["products"][0]["product_name"] == "Quest Chocolate Chip Cookie Dough Protein Bar"
    assert result["products"][0]["brands"] == "Quest"
    assert result["products"][0]["calories_kcal"] == 200


def test_spoonacular_search_recipe_returns_recipe_facts_and_consistency_warnings(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": 1,
                        "title": "Creamy Curry Chicken With Yellow Rice",
                        "image": None,
                        "nutrition": {
                            "nutrients": [
                                {"name": "Calories", "amount": 335.39},
                                {"name": "Protein", "amount": 3.67},
                                {"name": "Carbohydrates", "amount": 17.84},
                                {"name": "Fat", "amount": 30.09},
                            ]
                        },
                    }
                ]
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            return _FakeResponse()

    monkeypatch.setenv("SPOONACULAR_API_KEY", "test-key")
    monkeypatch.setattr("app.tools.nutrition_http_tools.httpx.Client", _FakeClient)

    result = spoonacular_search_recipe.func(query="chicken curry rice", number=3)

    assert "query_context" not in result
    assert result["results"][0]["title"] == "Creamy Curry Chicken With Yellow Rice"
    assert result["results"][0]["serving_basis"] == "recipe_serving"
    assert result["results"][0]["candidate_warnings"] == []
    assert result["results"][0]["internal_consistency_warning"] is None


def test_usda_search_foods_returns_unfiltered_provider_row_for_restaurant_named_query(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "foods": [
                    {
                        "fdcId": 2,
                        "description": "Turkey sandwich with lettuce and tomato",
                        "dataType": "Survey (FNDDS)",
                        "brandName": None,
                        "servingSize": 1,
                        "servingSizeUnit": "sandwich",
                        "foodNutrients": [
                            {"nutrientName": "Energy", "value": 147},
                            {"nutrientName": "Protein", "value": 9.1},
                            {"nutrientName": "Carbohydrate, by difference", "value": 22.4},
                            {"nutrientName": "Total lipid (fat)", "value": 2.3},
                        ],
                    }
                ],
                "totalHits": 1,
            }

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, params=None, json=None):
            return _FakeResponse()

    monkeypatch.setenv("USDA_API_KEY", "test-key")
    monkeypatch.setattr("app.tools.nutrition_http_tools.httpx.Client", _FakeClient)

    result = usda_search_foods.func(query="Subway 6-inch turkey sandwich", page_size=3)

    assert "query_context" not in result
    assert result["foods"][0]["description"] == "Turkey sandwich with lettuce and tomato"
    assert result["foods"][0]["calories_kcal"] == 147
    assert result["foods"][0]["protein_g"] == 9.1
