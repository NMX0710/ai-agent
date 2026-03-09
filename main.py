import base64
import hashlib
import hmac
import html
import logging
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from app.routers import sample
from app.recipe_app import RecipeApp
from app.settings import (
    SMS_ALLOWLIST_RAW,
    SMS_ENFORCE_TWILIO_SIGNATURE,
    SMS_TWILIO_AUTH_TOKEN,
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

# Jinja templates (web UI)
templates = Jinja2Templates(directory="app/templates")

# Mount static assets and include API routers
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(sample.router, prefix="/api")

_SMS_CACHE_TTL_SECONDS = 24 * 60 * 60
_sms_sid_cache: dict[str, tuple[float, str]] = {}
_TG_CACHE_TTL_SECONDS = 24 * 60 * 60
_tg_update_cache: dict[int, float] = {}


# ------------------------------------------------------------
# Local/Web chat endpoint
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    chat_id: str
    message: str
    user_id: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """Internal endpoint for local/Web chat usage."""
    return {"response": await recipe_app.chat(req.chat_id, req.message, user_id=req.user_id)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the interactive web chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


def _normalize_phone(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch in "+0123456789")


def _build_allowlist() -> set[str]:
    entries = [
        _normalize_phone(item.strip())
        for item in SMS_ALLOWLIST_RAW.split(",")
        if item.strip()
    ]
    return {item for item in entries if item}


SMS_ALLOWLIST = _build_allowlist()


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


def _build_twiml(message: str) -> str:
    safe = html.escape(message or "")
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe}</Message></Response>'


def _prune_sms_cache(now_ts: float) -> None:
    expired_keys = [
        sid for sid, (ts, _) in _sms_sid_cache.items()
        if now_ts - ts > _SMS_CACHE_TTL_SECONDS
    ]
    for sid in expired_keys:
        _sms_sid_cache.pop(sid, None)


def _prune_tg_cache(now_ts: float) -> None:
    expired = [
        update_id for update_id, ts in _tg_update_cache.items()
        if now_ts - ts > _TG_CACHE_TTL_SECONDS
    ]
    for update_id in expired:
        _tg_update_cache.pop(update_id, None)


def _twilio_signature_valid(url: str, form_data: dict[str, str], signature: str) -> bool:
    if not SMS_TWILIO_AUTH_TOKEN:
        return False

    data = url + "".join(f"{k}{form_data[k]}" for k in sorted(form_data))
    digest = hmac.new(
        SMS_TWILIO_AUTH_TOKEN.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")


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
    message = payload.get("message") or payload.get("edited_message")

    if not isinstance(update_id, int) or not isinstance(message, dict):
        return JSONResponse(content={"ok": True, "ignored": True})

    now_ts = time.time()
    _prune_tg_cache(now_ts)
    if update_id in _tg_update_cache:
        return JSONResponse(content={"ok": True, "duplicate": True})

    _tg_update_cache[update_id] = now_ts

    from_obj = message.get("from") or {}
    chat_obj = message.get("chat") or {}
    text = (message.get("text") or "").strip()
    message_id = message.get("message_id")

    from_id = from_obj.get("id")
    chat_id = chat_obj.get("id")
    if not isinstance(from_id, int) or not isinstance(chat_id, int):
        return JSONResponse(content={"ok": True, "ignored": True})

    if TELEGRAM_ALLOWLIST and from_id not in TELEGRAM_ALLOWLIST:
        await _telegram_send_text(chat_id, "This Telegram account is not authorized.", message_id)
        return JSONResponse(content={"ok": True, "blocked": True})

    if not text:
        await _telegram_send_text(chat_id, "Please send a text message with your diet request.", message_id)
        return JSONResponse(content={"ok": True, "ignored": True})

    user_id = f"tg:{from_id}"
    recipe_chat_id = f"tg:{chat_id}"

    logging.info("[Telegram] from_id=%s chat_id=%s update_id=%s text=%s", from_id, chat_id, update_id, text)
    answer = await recipe_app.chat(chat_id=recipe_chat_id, message=text, user_id=user_id)
    await _telegram_send_text(chat_id, answer, message_id)
    return JSONResponse(content={"ok": True})


@app.post("/webhooks/twilio/sms")
async def twilio_sms_webhook(request: Request):
    """
    Twilio inbound SMS webhook endpoint.

    Security and routing strategy:
    - Optional signature verification via SMS_ENFORCE_TWILIO_SIGNATURE
    - Allowlist-based sender filtering
    - per-peer session mapping: one session per sender-recipient pair
    """
    form = await request.form()
    form_data = {k: str(v) for k, v in form.items()}

    signature = request.headers.get("X-Twilio-Signature", "")
    if SMS_ENFORCE_TWILIO_SIGNATURE:
        if not _twilio_signature_valid(str(request.url), form_data, signature):
            logging.warning("[SMS] Invalid Twilio signature.")
            return Response(
                content=_build_twiml("Unauthorized request."),
                media_type="application/xml",
                status_code=403,
            )

    from_number_raw = form_data.get("From", "")
    to_number_raw = form_data.get("To", "")
    message_sid = form_data.get("MessageSid", "")
    body = (form_data.get("Body", "") or "").strip()
    num_media = int(form_data.get("NumMedia", "0") or 0)

    from_number = _normalize_phone(from_number_raw)
    to_number = _normalize_phone(to_number_raw)

    if SMS_ALLOWLIST and from_number not in SMS_ALLOWLIST:
        logging.info("[SMS] Blocked sender not in allowlist: %s", from_number_raw)
        return Response(
            content=_build_twiml("This number is not authorized for this assistant."),
            media_type="application/xml",
            status_code=200,
        )

    now_ts = time.time()
    _prune_sms_cache(now_ts)
    if message_sid and message_sid in _sms_sid_cache:
        cached_xml = _sms_sid_cache[message_sid][1]
        return Response(content=cached_xml, media_type="application/xml", status_code=200)

    if num_media > 0:
        xml = _build_twiml(
            "Image meal logging is not enabled yet. Please send text for now."
        )
        if message_sid:
            _sms_sid_cache[message_sid] = (now_ts, xml)
        return Response(content=xml, media_type="application/xml", status_code=200)

    if not body:
        xml = _build_twiml("Please send a text message with your diet request.")
        if message_sid:
            _sms_sid_cache[message_sid] = (now_ts, xml)
        return Response(content=xml, media_type="application/xml", status_code=200)

    user_id = f"sms:{from_number}"
    chat_id = f"sms:{from_number}->{to_number or 'unknown-recipient'}"

    logging.info(
        "[SMS] from=%s to=%s sid=%s body=%s",
        from_number,
        to_number,
        message_sid,
        body,
    )

    response_text = await recipe_app.chat(
        chat_id=chat_id,
        message=body,
        user_id=user_id,
    )
    xml = _build_twiml(response_text)
    if message_sid:
        _sms_sid_cache[message_sid] = (now_ts, xml)
    return Response(content=xml, media_type="application/xml", status_code=200)


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
