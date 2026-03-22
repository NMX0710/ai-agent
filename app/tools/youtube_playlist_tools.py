from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_core.tools import tool

from app.observability import trace_log
from app.settings import YOUTUBE_API_KEY, YOUTUBE_PLAYLIST_ID, YOUTUBE_PLAYLIST_MAX_ITEMS


YOUTUBE_PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _tokenize(value: Any) -> list[str]:
    return _TOKEN_RE.findall(_normalize_text(value))


def _score_video_match(query: str, title: str, description: str) -> int:
    normalized_query = _normalize_text(query)
    title_text = _normalize_text(title)
    description_text = _normalize_text(description)
    query_tokens = _tokenize(query)
    title_tokens = set(_tokenize(title))
    description_tokens = set(_tokenize(description))

    score = 0
    if normalized_query and normalized_query in title_text:
        score += 12
    if normalized_query and normalized_query in description_text:
        score += 6

    for token in query_tokens:
        if token in title_tokens:
            score += 4
        if token in description_tokens:
            score += 2

    return score


def _extract_video_row(item: dict[str, Any]) -> dict[str, Any] | None:
    snippet = item.get("snippet")
    if not isinstance(snippet, dict):
        return None

    resource = snippet.get("resourceId")
    if not isinstance(resource, dict):
        return None

    video_id = resource.get("videoId")
    title = snippet.get("title")
    if not isinstance(video_id, str) or not isinstance(title, str):
        return None
    if title in {"Deleted video", "Private video"}:
        return None

    description = snippet.get("description") if isinstance(snippet.get("description"), str) else ""
    return {
        "video_id": video_id,
        "title": title,
        "description": description,
        "channel_title": snippet.get("videoOwnerChannelTitle") or snippet.get("channelTitle"),
        "published_at": snippet.get("publishedAt"),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "source_kind": "playlist",
    }


def _fetch_playlist_items() -> list[dict[str, Any]] | dict[str, str]:
    if not YOUTUBE_API_KEY:
        return {"error": "missing YOUTUBE_API_KEY"}
    if not YOUTUBE_PLAYLIST_ID:
        return {"error": "missing YOUTUBE_PLAYLIST_ID"}

    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    remaining = max(1, YOUTUBE_PLAYLIST_MAX_ITEMS)

    try:
        with httpx.Client(timeout=30.0) as client:
            while remaining > 0:
                page_size = min(50, remaining)
                params = {
                    "part": "snippet",
                    "playlistId": YOUTUBE_PLAYLIST_ID,
                    "maxResults": page_size,
                    "key": YOUTUBE_API_KEY,
                }
                if page_token:
                    params["pageToken"] = page_token

                resp = client.get(YOUTUBE_PLAYLIST_ITEMS_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()

                for item in payload.get("items", []):
                    if isinstance(item, dict):
                        row = _extract_video_row(item)
                        if row:
                            rows.append(row)

                remaining = max(0, YOUTUBE_PLAYLIST_MAX_ITEMS - len(rows))
                page_token = payload.get("nextPageToken")
                if not isinstance(page_token, str) or not page_token:
                    break
    except Exception as exc:
        return {"error": f"youtube playlist request failed: {exc}"}

    return rows


def _extract_search_row(item: dict[str, Any]) -> dict[str, Any] | None:
    identifier = item.get("id")
    snippet = item.get("snippet")
    if not isinstance(identifier, dict) or not isinstance(snippet, dict):
        return None

    video_id = identifier.get("videoId")
    title = snippet.get("title")
    if not isinstance(video_id, str) or not isinstance(title, str):
        return None

    description = snippet.get("description") if isinstance(snippet.get("description"), str) else ""
    return {
        "video_id": video_id,
        "title": title,
        "description": description,
        "channel_title": snippet.get("channelTitle"),
        "published_at": snippet.get("publishedAt"),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "source_kind": "youtube_search",
    }


def _search_youtube_videos(query: str, max_results: int) -> list[dict[str, Any]] | dict[str, str]:
    if not YOUTUBE_API_KEY:
        return {"error": "missing YOUTUBE_API_KEY"}

    params = {
        "part": "snippet",
        "q": query,
        "maxResults": max(1, min(max_results, 10)),
        "type": "video",
        "key": YOUTUBE_API_KEY,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(YOUTUBE_SEARCH_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        return {"error": f"youtube search request failed: {exc}"}

    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if isinstance(item, dict):
            row = _extract_search_row(item)
            if row:
                rows.append(row)
    return rows


@tool(
    description=(
        "Search the configured YouTube recipe playlist for meal ideas, recipe inspiration, and meal-prep videos. "
        "Use this when the user wants suggestions for what to cook or eat, recipe references, or a weekly prep plan. "
        "Return playlist videos only, not general web or nutrition estimates."
    )
)
def search_youtube_playlist_recipes(query: str, max_results: int = 3) -> dict[str, Any]:
    capped_results = max(1, min(max_results, 5))
    trace_log(
        "ToolCall",
        "search_youtube_playlist_recipes invoked",
        {"query": query, "max_results": capped_results},
    )

    fetched = _fetch_playlist_items()
    if isinstance(fetched, dict):
        return fetched

    scored: list[tuple[int, dict[str, Any]]] = []
    for row in fetched:
        score = _score_video_match(query, row["title"], row["description"])
        if score <= 0:
            continue
        row_with_score = dict(row)
        row_with_score["match_score"] = score
        scored.append((score, row_with_score))

    scored.sort(key=lambda item: (item[0], item[1]["published_at"] or ""), reverse=True)
    playlist_results = [row for _, row in scored[:capped_results]]
    limited = list(playlist_results)

    if len(limited) < capped_results:
        fallback = _search_youtube_videos(query, capped_results)
        if isinstance(fallback, dict):
            trace_log(
                "ToolResult",
                "search_youtube_playlist_recipes fallback failed",
                {"query": query, "error": fallback["error"]},
            )
        else:
            seen_video_ids = {row["video_id"] for row in limited}
            fallback_scored: list[tuple[int, dict[str, Any]]] = []
            for row in fallback:
                if row["video_id"] in seen_video_ids:
                    continue
                score = _score_video_match(query, row["title"], row["description"])
                if score <= 0:
                    continue
                row_with_score = dict(row)
                row_with_score["match_score"] = score
                fallback_scored.append((score, row_with_score))

            fallback_scored.sort(key=lambda item: (item[0], item[1]["published_at"] or ""), reverse=True)
            for _, row in fallback_scored:
                if len(limited) >= capped_results:
                    break
                limited.append(row)

    trace_log(
        "ToolResult",
        "search_youtube_playlist_recipes returned",
        {
            "query": query,
            "count": len(limited),
            "preview": [
                {
                    "title": row["title"],
                    "url": row["url"],
                    "match_score": row["match_score"],
                    "source_kind": row["source_kind"],
                }
                for row in limited
            ],
        },
    )

    return {
        "results": limited,
        "count": len(limited),
        "source": "youtube_playlist_with_search_fallback",
        "playlist_id": YOUTUBE_PLAYLIST_ID,
        "playlist_count": len(playlist_results),
        "used_general_search_fallback": any(row["source_kind"] == "youtube_search" for row in limited),
    }
