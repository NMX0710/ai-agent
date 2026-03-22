from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

from app.settings import YOUTUBE_SUMMARY_MODEL


def _build_transcript_excerpt(segments: list[dict[str, Any]], limit_chars: int = 8000) -> str:
    chunks: list[str] = []
    total = 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        if total + len(text) > limit_chars:
            remaining = max(0, limit_chars - total)
            if remaining > 0:
                chunks.append(text[:remaining])
            break
        chunks.append(text)
        total += len(text)
    return "\n".join(chunks).strip()


def summarize_recipe_video(
    *,
    title: str,
    description: str,
    transcript_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    transcript_excerpt = _build_transcript_excerpt(transcript_segments)
    if not transcript_excerpt:
        fallback_summary = " ".join(description.split())[:400].strip()
        return {
            "summary": fallback_summary or title,
            "summary_source": "description_fallback",
            "transcript_excerpt": "",
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback_summary = transcript_excerpt[:400].strip()
        return {
            "summary": fallback_summary or title,
            "summary_source": "transcript_excerpt_fallback",
            "transcript_excerpt": transcript_excerpt,
        }

    model = ChatOpenAI(
        model=YOUTUBE_SUMMARY_MODEL,
        api_key=api_key,
        temperature=0.2,
    )
    prompt = (
        "You summarize cooking and meal-prep videos for retrieval.\n"
        "Write a compact summary in at most 3 sentences.\n"
        "Include the main dish or meal idea, the style or use case, and what makes it useful.\n"
        "Do not mention timestamps. Do not hallucinate details not present in the input.\n\n"
        f"Title: {title}\n"
        f"Description: {description}\n"
        "Transcript excerpt:\n"
        f"{transcript_excerpt}"
    )
    response = model.invoke(prompt)
    summary = str(getattr(response, "content", "") or "").strip()
    if not summary:
        summary = transcript_excerpt[:400].strip() or title

    return {
        "summary": summary,
        "summary_source": "llm_transcript_summary",
        "transcript_excerpt": transcript_excerpt,
    }
