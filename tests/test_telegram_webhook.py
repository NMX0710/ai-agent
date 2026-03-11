import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

import main


@pytest.fixture(autouse=True)
def _reset_telegram_runtime(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_WEBHOOK_SECRET", "")
    monkeypatch.setattr(main, "_tg_update_cache", {})


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


def test_telegram_photo_success_to_governance(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    sent = {}
    captured = {}
    chat_calls = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        sent["chat_id"] = chat_id
        sent["text"] = text
        sent["reply_to"] = reply_to_message_id

    async def fake_download_photo(photo_sizes):
        return {
            "file_id": "abc-file-id",
            "file_unique_id": "abc-unique-id",
            "file_path": "photos/demo.jpg",
            "mime_type": "image/jpeg",
            "file_size_bytes": 1234,
            "sha256": "f" * 64,
        }

    async def fake_process(event):
        captured["source"] = event.source
        captured["user_id"] = event.user_id
        captured["chat_id"] = event.chat_id
        captured["update_id"] = event.update_id
        captured["message_id"] = event.message_id
        captured["caption"] = event.caption
        captured["file_id"] = event.file_id
        return {"status": "accepted", "tracking_id": "tg-999"}

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        chat_calls["chat_id"] = chat_id
        chat_calls["message"] = message
        chat_calls["user_id"] = user_id
        return "Photo analyzed"

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main, "_telegram_download_photo_as_bytes", fake_download_photo)
    monkeypatch.setattr(main, "_process_telegram_photo_event", fake_process)
    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)

    payload = {
        "update_id": 105,
        "message": {
            "message_id": 36,
            "caption": "lunch today",
            "from": {"id": 999000},
            "chat": {"id": 555111},
            "photo": [
                {"file_id": "small", "file_unique_id": "uniq1", "width": 100, "height": 100},
                {"file_id": "large", "file_unique_id": "uniq2", "width": 1000, "height": 1000},
            ],
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert response.json().get("photo_processed") is True
    assert captured["source"] == "telegram"
    assert captured["user_id"] == "tg:999000"
    assert captured["chat_id"] == "tg:555111"
    assert captured["update_id"] == 105
    assert captured["message_id"] == 36
    assert captured["caption"] == "lunch today"
    assert captured["file_id"] == "abc-file-id"
    assert chat_calls["user_id"] == "tg:999000"
    assert chat_calls["chat_id"] == "tg:555111"
    assert "tracking_id: tg-999" in chat_calls["message"]
    assert sent["chat_id"] == 555111
    assert sent["text"] == "Photo analyzed"
    assert sent["reply_to"] == 36


def test_telegram_photo_download_failure(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    sent = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        sent["chat_id"] = chat_id
        sent["text"] = text
        sent["reply_to"] = reply_to_message_id

    async def fake_download_photo(photo_sizes):
        return None

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main, "_telegram_download_photo_as_bytes", fake_download_photo)

    payload = {
        "update_id": 106,
        "message": {
            "message_id": 37,
            "from": {"id": 999000},
            "chat": {"id": 555111},
            "photo": [{"file_id": "small", "file_unique_id": "uniq1", "width": 100, "height": 100}],
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert response.json().get("photo_processed") is False
    assert sent["chat_id"] == 555111
    assert "download failed" in sent["text"].lower()
    assert sent["reply_to"] == 37


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


def test_telegram_callback_confirm_success(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    sent = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        sent["chat_id"] = chat_id
        sent["text"] = text
        sent["reply_to"] = reply_to_message_id

    async def fake_answer(callback_query_id: str, text: str | None = None):
        sent["callback_query_id"] = callback_query_id
        sent["callback_text"] = text

    def fake_commit(*, draft_id: str, user_id: str, confirmed: bool):
        assert draft_id == "draft-1"
        assert user_id == "tg:999000"
        assert confirmed is True
        return {"ok": True, "status": "synced", "draft_id": draft_id}

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main, "_telegram_answer_callback_query", fake_answer)
    monkeypatch.setattr(main, "commit_meal_log_draft", fake_commit)

    payload = {
        "update_id": 120,
        "callback_query": {
            "id": "cbq-1",
            "from": {"id": 999000},
            "data": "meal_confirm:draft-1",
            "message": {"message_id": 50, "chat": {"id": 555111}},
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert response.json().get("callback_processed") is True
    assert sent["chat_id"] == 555111
    assert "written to apple health flow" in sent["text"].lower()
    assert sent["callback_query_id"] == "cbq-1"
