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
    assert result["query_context"]["food_form"] == "sushi_roll"
    assert result["query_context"]["likely_full_serving_request"] is True
    assert result["foods"][0]["is_composed_dish"] is True
    assert result["foods"][0]["serving_basis_unclear"] is True
    assert result["foods"][0]["likely_per_100g_or_small_portion"] is True
    assert result["foods"][0]["not_recommended_for_full_serving_estimate"] is True
    assert any(
        "Do not use this USDA row as the final answer" in warning
        for warning in result["foods"][0]["candidate_warnings"]
    )


def test_openfoodfacts_search_products_reports_basis_and_brand_match(monkeypatch):
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

    assert result["query_context"]["food_form"] == "protein_bar"
    assert result["query_context"]["likely_branded_packaged_food"] is True
    assert result["products"][0]["nutrition_basis"] == "serving"
    assert result["products"][0]["brand_match_confident"] is True
    assert "quest" in result["products"][0]["brand_match_tokens"]
    assert result["products"][0]["exact_name_match"] is True
    assert result["products"][0]["not_recommended_for_full_serving_estimate"] is False


def test_spoonacular_search_recipe_marks_weak_curry_candidate_not_recommended(monkeypatch):
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

    assert result["query_context"]["food_form"] == "curry_rice"
    assert result["results"][0]["dish_form_match"] is True
    assert result["results"][0]["not_recommended_for_full_serving_estimate"] is True
    assert any(
        "too small or unbalanced for a typical full-serving curry rice" in warning
        for warning in result["results"][0]["candidate_warnings"]
    )


def test_usda_search_foods_marks_generic_candidate_as_bad_for_restaurant_query(monkeypatch):
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

    assert result["query_context"]["likely_restaurant_or_menu_item"] is True
    assert result["foods"][0]["brand_match_confident"] is False
    assert result["foods"][0]["restaurant_query_mismatch"] is True
    assert result["foods"][0]["not_recommended_for_full_serving_estimate"] is True
    assert result["foods"][0]["calories_kcal"] is None
    assert result["foods"][0]["generic_reference_nutrition"]["calories_kcal"] == 147
