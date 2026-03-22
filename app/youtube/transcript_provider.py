from __future__ import annotations

from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi


def fetch_video_transcript(video_id: str, languages: list[str] | None = None) -> dict[str, Any]:
    preferred_languages = languages or ["en", "en-US", "zh-Hans", "zh-Hant", "zh"]
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=preferred_languages)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "segments": []}

    segments = []
    for item in transcript:
        text = getattr(item, "text", None)
        start = getattr(item, "start", None)
        duration = getattr(item, "duration", None)
        if not text:
            continue
        segments.append(
            {
                "text": str(text).strip(),
                "start": float(start) if isinstance(start, (int, float)) else None,
                "duration": float(duration) if isinstance(duration, (int, float)) else None,
            }
        )

    return {"ok": bool(segments), "segments": segments, "error": None if segments else "empty transcript"}
