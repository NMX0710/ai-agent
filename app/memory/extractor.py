import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "were",
    "have", "has", "had", "from", "into", "about", "please", "help", "want", "need",
    "then", "also", "just", "very", "some", "more", "than", "what", "when", "where",
    "how", "can", "could", "would", "should", "我", "你", "我们", "然后", "这个", "那个",
    "一下", "可以", "需要", "就是", "一个", "现在", "还是", "如果", "因为", "所以", "但是",
}


@dataclass
class MemoryCandidate:
    memory_key: str
    summary: str
    keywords: List[str]


def _extract_keywords(text: str, max_items: int = 12) -> List[str]:
    tokens = _TOKEN_PATTERN.findall(text or "")
    normalized: List[str] = []
    for token in tokens:
        normalized_token = token.lower() if token.isascii() else token
        if len(normalized_token) < 2:
            continue
        if normalized_token in _STOPWORDS:
            continue
        normalized.append(normalized_token)

    counts = Counter(normalized)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_items]]


def build_memory_candidate(user_message: str, assistant_message: str) -> Optional[MemoryCandidate]:
    keywords = _extract_keywords(f"{user_message}\n{assistant_message}")
    if not keywords:
        return None

    user_text = (user_message or "").strip().replace("\n", " ")
    assistant_text = (assistant_message or "").strip().replace("\n", " ")

    summary = (
        f"User preference/request: {user_text[:200]} | "
        f"Assistant suggestion: {assistant_text[:260]}"
    )
    key_source = "|".join(keywords[:6]) or summary[:120]
    memory_key = hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:24]

    return MemoryCandidate(
        memory_key=memory_key,
        summary=summary,
        keywords=keywords,
    )
