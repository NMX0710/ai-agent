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


def test_sms_blocked_by_allowlist(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "SMS_ALLOWLIST", {"+15550001111"})

    response = client.post(
        "/webhooks/twilio/sms",
        data={
            "From": "+15550002222",
            "To": "+15550009999",
            "Body": "hello",
            "MessageSid": "SM_DENIED_1",
            "NumMedia": "0",
        },
    )

    assert response.status_code == 200
    assert "not authorized" in response.text.lower()


def test_sms_chat_success(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "SMS_ALLOWLIST", {"+15550001111"})

    captured = {}

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        captured["chat_id"] = chat_id
        captured["message"] = message
        captured["user_id"] = user_id
        return "Test reply"

    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)

    response = client.post(
        "/webhooks/twilio/sms",
        data={
            "From": "+15550001111",
            "To": "+15550009999",
            "Body": "plan dinner",
            "MessageSid": "SM_OK_1",
            "NumMedia": "0",
        },
    )

    assert response.status_code == 200
    assert "<Message>Test reply</Message>" in response.text
    assert captured["user_id"] == "sms:+15550001111"
    assert captured["message"] == "plan dinner"
    assert captured["chat_id"] == "sms:+15550001111->+15550009999"


def test_sms_duplicate_message_sid_uses_cache(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "SMS_ALLOWLIST", {"+15550001111"})
    monkeypatch.setattr(main, "_sms_sid_cache", {})

    call_count = {"n": 0}

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        call_count["n"] += 1
        return "Cached reply"

    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)

    payload = {
        "From": "+15550001111",
        "To": "+15550009999",
        "Body": "same message",
        "MessageSid": "SM_CACHE_1",
        "NumMedia": "0",
    }

    first = client.post("/webhooks/twilio/sms", data=payload)
    second = client.post("/webhooks/twilio/sms", data=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert "Cached reply" in first.text
    assert "Cached reply" in second.text
    assert call_count["n"] == 1
