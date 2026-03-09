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


def test_telegram_blocked_by_allowlist(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {123456})

    sent = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        sent["chat_id"] = chat_id
        sent["text"] = text
        sent["reply_to"] = reply_to_message_id

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)

    payload = {
        "update_id": 101,
        "message": {
            "message_id": 33,
            "text": "hello",
            "from": {"id": 999000},
            "chat": {"id": 555111},
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert response.json().get("blocked") is True
    assert sent["chat_id"] == 555111
    assert "not authorized" in sent["text"].lower()


def test_telegram_chat_success(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    sent = {}
    captured = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        sent["chat_id"] = chat_id
        sent["text"] = text
        sent["reply_to"] = reply_to_message_id

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        captured["chat_id"] = chat_id
        captured["message"] = message
        captured["user_id"] = user_id
        return "TG reply"

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)

    payload = {
        "update_id": 102,
        "message": {
            "message_id": 34,
            "text": "plan dinner",
            "from": {"id": 999000},
            "chat": {"id": 555111},
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert response.json().get("ok") is True
    assert captured["user_id"] == "tg:999000"
    assert captured["chat_id"] == "tg:555111"
    assert captured["message"] == "plan dinner"
    assert sent["chat_id"] == 555111
    assert sent["text"] == "TG reply"
    assert sent["reply_to"] == 34


def test_telegram_duplicate_update_id(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    call_count = {"n": 0}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        call_count["n"] += 1

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        return "TG reply"

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)

    payload = {
        "update_id": 103,
        "message": {
            "message_id": 35,
            "text": "same",
            "from": {"id": 999000},
            "chat": {"id": 555111},
        },
    }
    first = client.post("/webhooks/telegram", json=payload)
    second = client.post("/webhooks/telegram", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json().get("duplicate") is True
    assert call_count["n"] == 1


def test_telegram_secret_validation(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_WEBHOOK_SECRET", "my-secret")

    payload = {"update_id": 104, "message": {"from": {"id": 1}, "chat": {"id": 2}, "text": "x"}}
    response = client.post("/webhooks/telegram", json=payload)
    assert response.status_code == 403
