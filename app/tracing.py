import json
import logging
import os
from contextvars import ContextVar, Token
from typing import Any

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def _env_enabled(name: str, default: str = "0") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def is_terminal_trace_enabled() -> bool:
    return _env_enabled("AGENT_TRACE_TERMINAL", "0")


def is_full_trace_enabled() -> bool:
    return _env_enabled("AGENT_TRACE_FULL", "0")


def set_trace_id(trace_id: str) -> Token:
    return _trace_id_var.set(trace_id)


def reset_trace_id(token: Token) -> None:
    _trace_id_var.reset(token)


def get_trace_id() -> str:
    return _trace_id_var.get()


def _preview(value: Any, limit: int = 800) -> str:
    try:
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = str(value)
    except Exception:
        text = repr(value)

    if is_full_trace_enabled() or len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def trace_log(stage: str, message: str, payload: Any = None) -> None:
    if not is_terminal_trace_enabled():
        return

    trace_id = get_trace_id()
    if payload is None:
        logging.info("[Trace][%s][%s] %s", trace_id, stage, message)
        return
    logging.info(
        "[Trace][%s][%s] %s | payload=%s",
        trace_id,
        stage,
        message,
        _preview(payload),
    )
