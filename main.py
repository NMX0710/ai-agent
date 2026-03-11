import hashlib
import hmac
import logging
import re
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from app.governance import ImageGovernanceEvent, process_image_event
from app.nutrition.meal_log_service import (
    cancel_meal_log_draft,
    commit_meal_log_draft,
    latest_pending_draft_for_chat,
    list_pending_apple_health_writes,
    mark_confirmation_prompted,
    report_apple_health_write_result,
)
from app.routers import sample
from app.recipe_app import RecipeApp
from app.settings import (
    APPLE_HEALTH_BRIDGE_TOKEN,
    TELEGRAM_ALLOWLIST_RAW,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_WEBHOOK_SECRET,
)

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

app = FastAPI()

# Create a singleton RecipeApp instance shared across requests
recipe_app = RecipeApp()


@app.on_event("shutdown")
async def _shutdown_recipe_app() -> None:
    await recipe_app.close()

# Jinja templates (web UI)
templates = Jinja2Templates(directory="app/templates")

# Mount static assets and include API routers
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(sample.router, prefix="/api")

_TG_CACHE_TTL_SECONDS = 24 * 60 * 60
_tg_update_cache: dict[int, float] = {}


# ------------------------------------------------------------
# Local/Web chat endpoint
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    chat_id: str
    message: str
    user_id: str


class AppleHealthPendingRequest(BaseModel):
    user_id: str
    limit: int = 20


class AppleHealthWriteReportRequest(BaseModel):
    user_id: str
    draft_id: str
    success: bool
    external_id: str | None = None
    error: str | None = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """Internal endpoint for local/Web chat usage."""
    return {"response": await recipe_app.chat(req.chat_id, req.message, user_id=req.user_id)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the interactive web chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


def _build_telegram_allowlist() -> set[int]:
    out: set[int] = set()
    for raw in TELEGRAM_ALLOWLIST_RAW.split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            out.add(int(value))
        except ValueError:
            logging.warning("[Telegram] Ignoring invalid allowlist entry: %s", value)
    return out


TELEGRAM_ALLOWLIST = _build_telegram_allowlist()


def _bridge_authorized(request: Request) -> bool:
    if not APPLE_HEALTH_BRIDGE_TOKEN:
        return True
    token = request.headers.get("X-Apple-Bridge-Token", "")
    return hmac.compare_digest(token, APPLE_HEALTH_BRIDGE_TOKEN)


def _prune_tg_cache(now_ts: float) -> None:
    expired = [
        update_id for update_id, ts in _tg_update_cache.items()
        if now_ts - ts > _TG_CACHE_TTL_SECONDS
    ]
    for update_id in expired:
        _tg_update_cache.pop(update_id, None)


async def _telegram_send_text(chat_id: int, text: str, reply_to_message_id: int | None = None) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logging.error("[Telegram] TELEGRAM_BOT_TOKEN is not configured.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logging.error("[Telegram] sendMessage failed: %s - %s", resp.status_code, resp.text)


async def _telegram_send_confirmation_prompt(
    chat_id: int,
    draft_id: str,
    text: str,
    reply_to_message_id: int | None = None,
) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logging.error("[Telegram] TELEGRAM_BOT_TOKEN is not configured.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "Confirm Write", "callback_data": f"meal_confirm:{draft_id}"},
                    {"text": "Cancel", "callback_data": f"meal_cancel:{draft_id}"},
                ]
            ]
        },
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logging.error("[Telegram] send confirmation prompt failed: %s - %s", resp.status_code, resp.text)


def _is_probably_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _format_meal_estimate_message(
    *,
    language_hint_text: str,
    energy_kcal: float,
    protein_g: float,
    carbs_g: float,
    fat_g: float,
) -> str:
    if _is_probably_chinese(language_hint_text):
        return (
            "当然可以，我先帮你做了估算：\n"
            f"- 热量约 {energy_kcal:.0f} kcal\n"
            f"- 蛋白质约 {protein_g:.1f} g\n"
            f"- 碳水约 {carbs_g:.1f} g\n"
            f"- 脂肪约 {fat_g:.1f} g\n\n"
            "你看这样可以吗？如果可以请点下方【确认保存】，我就写入 Apple Health。"
        )
    return (
        "Sure. Here is my estimate for this meal:\n"
        f"- Energy: ~{energy_kcal:.0f} kcal\n"
        f"- Protein: ~{protein_g:.1f} g\n"
        f"- Carbs: ~{carbs_g:.1f} g\n"
        f"- Fat: ~{fat_g:.1f} g\n\n"
        "Does this look right? If yes, tap Confirm Write below and I will sync it to Apple Health."
    )


async def _telegram_answer_callback_query(callback_query_id: str, text: str | None = None) -> None:
    if not TELEGRAM_BOT_TOKEN:
        logging.error("[Telegram] TELEGRAM_BOT_TOKEN is not configured.")
        return
    if not callback_query_id:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    payload: dict[str, Any] = {"callback_query_id": callback_query_id}
    if text:
        payload["text"] = text
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logging.error("[Telegram] answerCallbackQuery failed: %s - %s", resp.status_code, resp.text)


async def _maybe_prompt_meal_confirmation(
    *,
    chat_id: int,
    user_id: str,
    recipe_chat_id: str,
    reply_to_message_id: int | None,
    language_hint_text: str = "",
) -> bool:
    pending = latest_pending_draft_for_chat(user_id=user_id, chat_id=recipe_chat_id)
    if not pending or pending.confirmation_prompted:
        return False

    totals = pending.meal_log.nutrition_totals
    text = _format_meal_estimate_message(
        language_hint_text=language_hint_text or (pending.meal_log.raw_inputs.text if pending.meal_log.raw_inputs else ""),
        energy_kcal=totals.energy_kcal,
        protein_g=totals.protein_g,
        carbs_g=totals.carbs_g,
        fat_g=totals.fat_g,
    )
    await _telegram_send_confirmation_prompt(
        chat_id=chat_id,
        draft_id=pending.draft_id,
        text=text,
        reply_to_message_id=reply_to_message_id,
    )
    mark_confirmation_prompted(pending.draft_id)
    return True


def _guess_mime_type_from_path(file_path: str) -> str:
    low = (file_path or "").lower()
    if low.endswith(".jpg") or low.endswith(".jpeg"):
        return "image/jpeg"
    if low.endswith(".png"):
        return "image/png"
    if low.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def _select_best_photo_size(photo_sizes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not photo_sizes:
        return None

    def _score(item: dict[str, Any]) -> tuple[int, int]:
        file_size = item.get("file_size")
        if isinstance(file_size, int) and file_size > 0:
            return (file_size, 1)
        width = item.get("width") if isinstance(item.get("width"), int) else 0
        height = item.get("height") if isinstance(item.get("height"), int) else 0
        return (width * height, 0)

    ranked = sorted(
        (item for item in photo_sizes if isinstance(item, dict)),
        key=_score,
        reverse=True,
    )
    return ranked[0] if ranked else None


async def _telegram_get_file_path(file_id: str) -> str | None:
    if not TELEGRAM_BOT_TOKEN or not file_id:
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
    payload = {"file_id": file_id}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logging.error("[Telegram] getFile failed: %s - %s", resp.status_code, resp.text)
            return None
        data = resp.json()
    if not data.get("ok"):
        logging.error("[Telegram] getFile returned non-ok payload: %s", data)
        return None

    result = data.get("result") or {}
    file_path = result.get("file_path")
    return file_path if isinstance(file_path, str) else None


async def _telegram_download_file_bytes(file_path: str) -> bytes | None:
    if not TELEGRAM_BOT_TOKEN or not file_path:
        return None

    url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code >= 400:
            logging.error("[Telegram] file download failed: %s - %s", resp.status_code, resp.text)
            return None
        return resp.content


async def _telegram_download_photo_as_bytes(photo_sizes: list[dict[str, Any]]) -> dict[str, Any] | None:
    chosen = _select_best_photo_size(photo_sizes)
    if not chosen:
        return None

    file_id = chosen.get("file_id")
    file_unique_id = chosen.get("file_unique_id")
    if not isinstance(file_id, str) or not isinstance(file_unique_id, str):
        return None

    file_path = await _telegram_get_file_path(file_id)
    if not file_path:
        return None

    image_bytes = await _telegram_download_file_bytes(file_path)
    if not image_bytes:
        return None

    return {
        "file_id": file_id,
        "file_unique_id": file_unique_id,
        "file_path": file_path,
        "mime_type": _guess_mime_type_from_path(file_path),
        "file_size_bytes": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }


async def _process_telegram_photo_event(event: ImageGovernanceEvent) -> dict[str, Any]:
    return await process_image_event(event)


@app.post("/webhooks/telegram")
async def telegram_webhook(request: Request):
    """
    Telegram inbound webhook endpoint.

    Behavior:
    - Optional webhook secret validation if TELEGRAM_WEBHOOK_SECRET is set
    - allowlist enforcement by numeric Telegram user id
    - per-peer chat routing: user_id=tg:<from.id>, chat_id=tg:<chat.id>
    - update_id deduplication to avoid duplicate replies on retries
    """
    if TELEGRAM_WEBHOOK_SECRET:
        secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(secret, TELEGRAM_WEBHOOK_SECRET):
            return JSONResponse(content={"ok": False, "error": "unauthorized"}, status_code=403)

    payload = await request.json()
    update_id = payload.get("update_id")
    callback_query = payload.get("callback_query")
    message = payload.get("message") or payload.get("edited_message")

    if not isinstance(update_id, int):
        return JSONResponse(content={"ok": True, "ignored": True})

    now_ts = time.time()
    _prune_tg_cache(now_ts)
    if update_id in _tg_update_cache:
        return JSONResponse(content={"ok": True, "duplicate": True})

    _tg_update_cache[update_id] = now_ts

    if isinstance(callback_query, dict):
        from_obj = callback_query.get("from") or {}
        callback_message = callback_query.get("message") or {}
        from_id = from_obj.get("id")
        callback_id = callback_query.get("id")
        data = (callback_query.get("data") or "").strip()
        chat_obj = callback_message.get("chat") or {}
        chat_id = chat_obj.get("id")
        message_id = callback_message.get("message_id")

        if not isinstance(from_id, int) or not isinstance(chat_id, int):
            return JSONResponse(content={"ok": True, "ignored": True})

        if TELEGRAM_ALLOWLIST and from_id not in TELEGRAM_ALLOWLIST:
            await _telegram_answer_callback_query(callback_id if isinstance(callback_id, str) else "", "Unauthorized")
            return JSONResponse(content={"ok": True, "blocked": True})

        user_id = f"tg:{from_id}"
        if data.startswith("meal_confirm:"):
            draft_id = data.split("meal_confirm:", 1)[1].strip()
            result = commit_meal_log_draft(draft_id=draft_id, user_id=user_id, confirmed=True)
            if result.get("ok"):
                text = "Meal log confirmed and queued for Apple Health sync."
                callback_text = "Confirmed"
            else:
                text = f"Write failed: {result.get('error', 'unknown_error')}"
                callback_text = "Write failed"
        elif data.startswith("meal_cancel:"):
            draft_id = data.split("meal_cancel:", 1)[1].strip()
            result = cancel_meal_log_draft(draft_id=draft_id, user_id=user_id)
            if result.get("ok"):
                text = "Meal log draft cancelled."
                callback_text = "Cancelled"
            else:
                text = f"Cancel failed: {result.get('error', 'unknown_error')}"
                callback_text = "Cancel failed"
        else:
            return JSONResponse(content={"ok": True, "ignored": True})

        await _telegram_answer_callback_query(callback_id if isinstance(callback_id, str) else "", callback_text)
        await _telegram_send_text(chat_id, text, message_id if isinstance(message_id, int) else None)
        return JSONResponse(content={"ok": True, "callback_processed": True})

    if not isinstance(message, dict):
        return JSONResponse(content={"ok": True, "ignored": True})

    from_obj = message.get("from") or {}
    chat_obj = message.get("chat") or {}
    text = (message.get("text") or "").strip()
    caption = (message.get("caption") or "").strip()
    photo_sizes = message.get("photo") if isinstance(message.get("photo"), list) else []
    message_id = message.get("message_id")
    event_ts = message.get("date") if isinstance(message.get("date"), int) else None

    from_id = from_obj.get("id")
    chat_id = chat_obj.get("id")
    if not isinstance(from_id, int) or not isinstance(chat_id, int):
        return JSONResponse(content={"ok": True, "ignored": True})

    if TELEGRAM_ALLOWLIST and from_id not in TELEGRAM_ALLOWLIST:
        await _telegram_send_text(chat_id, "This Telegram account is not authorized.", message_id)
        return JSONResponse(content={"ok": True, "blocked": True})

    user_id = f"tg:{from_id}"
    recipe_chat_id = f"tg:{chat_id}"

    if photo_sizes:
        try:
            image_meta = await _telegram_download_photo_as_bytes(photo_sizes)
            if not image_meta:
                await _telegram_send_text(
                    chat_id,
                    "Image received, but download failed. Please try sending the photo again.",
                    message_id,
                )
                return JSONResponse(content={"ok": True, "photo_processed": False})

            gov_event = ImageGovernanceEvent(
                source="telegram",
                user_id=user_id,
                chat_id=recipe_chat_id,
                update_id=update_id,
                message_id=message_id if isinstance(message_id, int) else None,
                event_ts=event_ts,
                caption=caption,
                file_id=image_meta["file_id"],
                file_unique_id=image_meta["file_unique_id"],
                file_path=image_meta["file_path"],
                mime_type=image_meta["mime_type"],
                file_size_bytes=image_meta["file_size_bytes"],
                sha256=image_meta["sha256"],
            )
            gov_result = await _process_telegram_photo_event(gov_event)
            tracking_id = gov_result.get("tracking_id", "n/a")
            photo_message = (
                "User sent a meal photo in Telegram.\n"
                f"tracking_id: {tracking_id}\n"
                f"caption: {caption or '(none)'}\n"
                "If user intent is to log a meal, prepare a meal log draft first. "
                "If missing dish details, ask a concise follow-up question."
            )
            answer = await recipe_app.chat(chat_id=recipe_chat_id, message=photo_message, user_id=user_id)
            prompted = await _maybe_prompt_meal_confirmation(
                chat_id=chat_id,
                user_id=user_id,
                recipe_chat_id=recipe_chat_id,
                reply_to_message_id=message_id if isinstance(message_id, int) else None,
                language_hint_text=caption,
            )
            if not prompted:
                await _telegram_send_text(chat_id, answer, message_id)
            return JSONResponse(content={"ok": True, "photo_processed": True, "tracking_id": tracking_id})
        except Exception:
            logging.exception("[Telegram] Photo processing failed. update_id=%s", update_id)
            await _telegram_send_text(
                chat_id,
                "Image processing failed on server side. Please try again later.",
                message_id,
            )
            return JSONResponse(content={"ok": True, "photo_processed": False})

    if not text:
        await _telegram_send_text(chat_id, "Please send a text message with your diet request.", message_id)
        return JSONResponse(content={"ok": True, "ignored": True})

    logging.info("[Telegram] from_id=%s chat_id=%s update_id=%s text=%s", from_id, chat_id, update_id, text)
    answer = await recipe_app.chat(chat_id=recipe_chat_id, message=text, user_id=user_id)
    prompted = await _maybe_prompt_meal_confirmation(
        chat_id=chat_id,
        user_id=user_id,
        recipe_chat_id=recipe_chat_id,
        reply_to_message_id=message_id if isinstance(message_id, int) else None,
        language_hint_text=text,
    )
    if not prompted:
        await _telegram_send_text(chat_id, answer, message_id)
    return JSONResponse(content={"ok": True})


@app.post("/integrations/apple-health/pending-writes")
async def apple_health_pending_writes(request: Request, body: AppleHealthPendingRequest):
    if not _bridge_authorized(request):
        return JSONResponse(content={"ok": False, "error": "unauthorized"}, status_code=403)
    rows = list_pending_apple_health_writes(user_id=body.user_id, limit=body.limit)
    return {"ok": True, "count": len(rows), "items": rows}


@app.post("/integrations/apple-health/write-result")
async def apple_health_write_result(request: Request, body: AppleHealthWriteReportRequest):
    if not _bridge_authorized(request):
        return JSONResponse(content={"ok": False, "error": "unauthorized"}, status_code=403)
    result = report_apple_health_write_result(
        draft_id=body.draft_id,
        user_id=body.user_id,
        success=body.success,
        external_id=body.external_id,
        error=body.error,
    )
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(content=result, status_code=status_code)


# ------------------------------------------------------------
# AgentCore-required endpoints
# ------------------------------------------------------------
@app.get("/ping")
def ping():
    """Health check endpoint for AgentCore Runtime."""
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(req: Request):
    """
    Unified invocation entrypoint for AgentCore Runtime.

    AgentCore wraps model inputs (e.g., "prompt") and sends JSON to this endpoint.
    Expected schema:
      body = {"input": {"prompt": "..."}}
    """
    body = await req.json()
    logging.info("[AgentCore] Received request payload: %s", body)

    prompt = (body.get("input") or {}).get("prompt", "")
    input_obj = body.get("input") or {}
    user_id = input_obj.get("user_id") or "agentcore-user"
    conversation_id = input_obj.get("conversation_id") or "agentcore-session"

    # Reuse the same chat logic as the local/Web endpoint.
    response_text = await recipe_app.chat(
        chat_id=conversation_id,
        message=prompt,
        user_id=user_id,
    )

    # Standardize the response schema for AgentCore.
    return JSONResponse(content={"output": response_text})
