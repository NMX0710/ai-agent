import sys
from pathlib import Path

from fastapi.testclient import TestClient

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

import main
from app.nutrition.draft_store import get_meal_draft_store
from app.nutrition.meal_log_service import commit_meal_log_draft, prepare_meal_log_draft
from app.nutrition.meallog import InputSource, NutritionTotals


def _reset_store():
    store = get_meal_draft_store()
    store._records.clear()  # test-only reset of singleton in-memory store


def test_pending_writes_endpoint_success(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "APPLE_HEALTH_BRIDGE_TOKEN", "")

    def fake_list(*, user_id: str, limit: int = 20, lease_seconds: int = 300):
        assert user_id == "tg:1"
        assert limit == 10
        assert lease_seconds == 120
        return [{"draft_id": "d1"}, {"draft_id": "d2"}]

    monkeypatch.setattr(main, "list_pending_apple_health_writes", fake_list)

    resp = client.post(
        "/integrations/apple-health/pending-writes",
        json={"user_id": "tg:1", "limit": 10, "lease_seconds": 120},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["count"] == 2


def test_pending_writes_endpoint_token_required(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "APPLE_HEALTH_BRIDGE_TOKEN", "secret-1")

    resp = client.post(
        "/integrations/apple-health/pending-writes",
        json={"user_id": "tg:1"},
    )

    assert resp.status_code == 403
    assert resp.json()["error"] == "unauthorized"


def test_write_result_endpoint_success(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "APPLE_HEALTH_BRIDGE_TOKEN", "")

    def fake_report(*, draft_id: str, user_id: str, success: bool, claim_token=None, external_id=None, error=None):
        assert draft_id == "d1"
        assert user_id == "tg:1"
        assert success is True
        assert claim_token == "claim-1"
        return {"ok": True, "status": "synced", "draft_id": draft_id}

    monkeypatch.setattr(main, "report_apple_health_write_result", fake_report)

    resp = client.post(
        "/integrations/apple-health/write-result",
        json={"user_id": "tg:1", "draft_id": "d1", "success": True, "claim_token": "claim-1"},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "synced"


def test_write_result_endpoint_failure_returns_400(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "APPLE_HEALTH_BRIDGE_TOKEN", "")

    monkeypatch.setattr(
        main,
        "report_apple_health_write_result",
        lambda **_: {"ok": False, "error": "draft_not_found"},
    )

    resp = client.post(
        "/integrations/apple-health/write-result",
        json={"user_id": "tg:1", "draft_id": "missing", "success": False},
    )

    assert resp.status_code == 400
    assert resp.json()["error"] == "draft_not_found"


def test_bridge_api_claims_once_and_accepts_idempotent_report(monkeypatch):
    _reset_store()
    client = TestClient(main.app)
    monkeypatch.setattr(main, "APPLE_HEALTH_BRIDGE_TOKEN", "")

    draft = prepare_meal_log_draft(
        user_id="tg:bridge",
        chat_id="tg:chat-bridge",
        input_source=InputSource.text,
        meal_description="rice bowl",
        nutrition_totals=NutritionTotals(energy_kcal=410, protein_g=22, carbs_g=44, fat_g=14),
        nutrition_source="test_source",
        nutrition_confidence=0.7,
    )
    commit_meal_log_draft(draft_id=draft["draft_id"], user_id="tg:bridge", confirmed=True)

    first_pending = client.post(
        "/integrations/apple-health/pending-writes",
        json={"user_id": "tg:bridge", "limit": 10, "lease_seconds": 300},
    )
    assert first_pending.status_code == 200
    first_items = first_pending.json()["items"]
    assert len(first_items) == 1
    claim_token = first_items[0]["claim_token"]
    assert isinstance(claim_token, str)

    second_pending = client.post(
        "/integrations/apple-health/pending-writes",
        json={"user_id": "tg:bridge", "limit": 10, "lease_seconds": 300},
    )
    assert second_pending.status_code == 200
    assert second_pending.json()["items"] == []

    synced = client.post(
        "/integrations/apple-health/write-result",
        json={
            "user_id": "tg:bridge",
            "draft_id": draft["draft_id"],
            "success": True,
            "claim_token": claim_token,
            "external_id": "hk-bridge-1",
        },
    )
    assert synced.status_code == 200
    assert synced.json()["status"] == "synced"

    duplicate = client.post(
        "/integrations/apple-health/write-result",
        json={
            "user_id": "tg:bridge",
            "draft_id": draft["draft_id"],
            "success": True,
            "claim_token": claim_token,
            "external_id": "hk-bridge-1",
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "synced"
