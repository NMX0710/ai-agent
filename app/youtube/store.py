from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.settings import YOUTUBE_CACHE_DIR


def _cache_path(video_id: str) -> Path:
    return YOUTUBE_CACHE_DIR / f"{video_id}.json"


def load_video_cache(video_id: str) -> dict[str, Any] | None:
    path = _cache_path(video_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_video_cache(video_id: str, payload: dict[str, Any]) -> None:
    YOUTUBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(video_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
