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


def test_pending_writes_endpoint_success(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "APPLE_HEALTH_BRIDGE_TOKEN", "")

    def fake_list(*, user_id: str, limit: int = 20):
        assert user_id == "tg:1"
        assert limit == 10
        return [{"draft_id": "d1"}, {"draft_id": "d2"}]

    monkeypatch.setattr(main, "list_pending_apple_health_writes", fake_list)

    resp = client.post(
        "/integrations/apple-health/pending-writes",
        json={"user_id": "tg:1", "limit": 10},
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

    def fake_report(*, draft_id: str, user_id: str, success: bool, external_id=None, error=None):
        assert draft_id == "d1"
        assert user_id == "tg:1"
        assert success is True
        return {"ok": True, "status": "synced", "draft_id": draft_id}

    monkeypatch.setattr(main, "report_apple_health_write_result", fake_report)

    resp = client.post(
        "/integrations/apple-health/write-result",
        json={"user_id": "tg:1", "draft_id": "d1", "success": True},
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
