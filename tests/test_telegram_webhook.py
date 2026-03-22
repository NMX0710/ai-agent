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
from app.nutrition.draft_store import MealDraftRecord
from app.nutrition.meallog import CanonicalMealLog, ConfidenceInfo, InputSource, MealLogStatus, NutritionTotals


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
    assert "queued for apple health sync" in sent["text"].lower()
    assert sent["callback_query_id"] == "cbq-1"


def test_telegram_record_flow_sends_single_estimate_confirmation(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    confirmation = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        raise AssertionError("raw text send should be suppressed when draft confirmation is available")

    async def fake_confirmation(chat_id: int, draft_id: str, text: str, reply_to_message_id=None):
        confirmation["chat_id"] = chat_id
        confirmation["draft_id"] = draft_id
        confirmation["text"] = text
        confirmation["reply_to"] = reply_to_message_id

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        return "agent raw response should be suppressed"

    meal_log = CanonicalMealLog(
        meal_id="4f52ba58-4dca-4f64-b115-bf00a264f6f0",
        user_id="tg:999000",
        consumed_at="2026-03-10T20:00:00-04:00",
        timezone="America/New_York",
        input_source=InputSource.text,
        status=MealLogStatus.draft,
        nutrition_totals=NutritionTotals(energy_kcal=600, protein_g=25, carbs_g=70, fat_g=20),
        confidence=ConfidenceInfo(overall=0.8),
    )
    pending = MealDraftRecord(
        draft_id="draft-xyz",
        user_id="tg:999000",
        chat_id="tg:555111",
        meal_log=meal_log,
        status="draft",
        created_at=meal_log.consumed_at,
        updated_at=meal_log.consumed_at,
        confirmation_prompted=False,
    )

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main, "_telegram_send_confirmation_prompt", fake_confirmation)
    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)
    monkeypatch.setattr(main, "latest_pending_draft_for_chat", lambda user_id, chat_id: pending)
    monkeypatch.setattr(main, "mark_confirmation_prompted", lambda draft_id: None)

    payload = {
        "update_id": 130,
        "message": {
            "message_id": 88,
            "text": "我晚上吃了意大利面 可以帮我记录吗",
            "from": {"id": 999000},
            "chat": {"id": 555111},
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert confirmation["draft_id"] == "draft-xyz"
    assert "热量约 600" in confirmation["text"]
    assert "确认保存" in confirmation["text"]
    assert "agent raw response should be suppressed" not in confirmation["text"]


def test_telegram_invalid_draft_does_not_suppress_agent_clarification(monkeypatch):
    client = TestClient(main.app)
    monkeypatch.setattr(main, "TELEGRAM_ALLOWLIST", {999000})
    monkeypatch.setattr(main, "_tg_update_cache", {})

    sent = {}

    async def fake_send(chat_id: int, text: str, reply_to_message_id=None):
        sent["chat_id"] = chat_id
        sent["text"] = text
        sent["reply_to"] = reply_to_message_id

    async def fake_confirmation(chat_id: int, draft_id: str, text: str, reply_to_message_id=None):
        raise AssertionError("confirmation prompt should not be sent for invalid draft")

    async def fake_chat(chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        return "请告诉我大致分量，我再帮你估算。"

    meal_log = CanonicalMealLog(
        meal_id="4f52ba58-4dca-4f64-b115-bf00a264f6f1",
        user_id="tg:999000",
        consumed_at="2026-03-10T20:00:00-04:00",
        timezone="America/New_York",
        input_source=InputSource.text,
        status=MealLogStatus.draft,
        nutrition_totals=NutritionTotals(energy_kcal=0, protein_g=0, carbs_g=0, fat_g=0),
        confidence=ConfidenceInfo(overall=0.8),
    )
    pending = MealDraftRecord(
        draft_id="draft-invalid",
        user_id="tg:999000",
        chat_id="tg:555111",
        meal_log=meal_log,
        status="draft",
        created_at=meal_log.consumed_at,
        updated_at=meal_log.consumed_at,
        confirmation_prompted=False,
    )

    monkeypatch.setattr(main, "_telegram_send_text", fake_send)
    monkeypatch.setattr(main, "_telegram_send_confirmation_prompt", fake_confirmation)
    monkeypatch.setattr(main.recipe_app, "chat", fake_chat)
    monkeypatch.setattr(main, "latest_pending_draft_for_chat", lambda user_id, chat_id: pending)
    monkeypatch.setattr(main, "mark_confirmation_prompted", lambda draft_id: None)

    payload = {
        "update_id": 131,
        "message": {
            "message_id": 89,
            "text": "我晚上吃了肉酱意大利面 可以帮我记录吗",
            "from": {"id": 999000},
            "chat": {"id": 555111},
        },
    }
    response = client.post("/webhooks/telegram", json=payload)

    assert response.status_code == 200
    assert sent["chat_id"] == 555111
    assert sent["text"] == "请告诉我大致分量，我再帮你估算。"
    assert sent["reply_to"] == 89


def test_split_telegram_youtube_messages_splits_body_and_urls():
    raw = (
        "Here are some high-protein lunch ideas.\n\n"
        "Video 1: Chicken rice bowl meal prep.\n"
        "Quick, high-protein lunch for weekdays.\n"
        "https://www.youtube.com/watch?v=abc123\n\n"
        "Video 2: Greek yogurt chicken salad.\n"
        "Easy cold lunch option.\n"
        "https://www.youtube.com/watch?v=def456"
    )

    messages = main._split_telegram_youtube_messages(raw)

    assert len(messages) == 2
    assert "Here are some high-protein lunch ideas." in messages[0]
    assert "Video 1: Chicken rice bowl meal prep." in messages[0]
    assert messages[0].endswith("https://www.youtube.com/watch?v=abc123")
    assert "Video 2: Greek yogurt chicken salad." in messages[1]
    assert messages[1].endswith("https://www.youtube.com/watch?v=def456")


def test_telegram_send_text_sends_split_youtube_messages(monkeypatch):
    captured = {"payloads": []}

    class _FakeResponse:
        status_code = 200
        text = "ok"

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None):
            captured["url"] = url
            captured["payloads"].append(json)
            return _FakeResponse()

    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(main.httpx, "AsyncClient", _FakeAsyncClient)

    import asyncio

    asyncio.run(
        main._telegram_send_text(
            555111,
            (
                "Here are some high-protein lunch ideas.\n\n"
                "Video 1 summary.\n"
                "https://www.youtube.com/watch?v=abc123\n\n"
                "Video 2 summary.\n"
                "https://www.youtube.com/watch?v=def456"
            ),
            34,
        )
    )

    assert len(captured["payloads"]) == 2
    assert "Here are some high-protein lunch ideas." in captured["payloads"][0]["text"]
    assert "Video 1 summary." in captured["payloads"][0]["text"]
    assert captured["payloads"][0]["reply_to_message_id"] == 34
    assert captured["payloads"][0]["text"].endswith("https://www.youtube.com/watch?v=abc123")
    assert captured["payloads"][0]["link_preview_options"]["url"] == "https://www.youtube.com/watch?v=abc123"
    assert "Video 2 summary." in captured["payloads"][1]["text"]
    assert "reply_to_message_id" not in captured["payloads"][1]
    assert captured["payloads"][1]["text"].endswith("https://www.youtube.com/watch?v=def456")
    assert captured["payloads"][1]["link_preview_options"]["url"] == "https://www.youtube.com/watch?v=def456"
