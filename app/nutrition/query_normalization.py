from __future__ import annotations

from dataclasses import dataclass
import re


_ZH_LEADING_PATTERNS = [
    r"^(?:请|帮我|麻烦你)?(?:帮我记录一下|帮我记一下|记录一下|记录下|记录|记一下|记下|记个|记到苹果健康里|记到apple health里)",
    r"^(?:我(?:今天|刚刚|刚才|晚上|中午|早上)?(?:吃了|喝了)|今晚吃了|今天吃了)",
]
_ZH_TRAILING_PATTERNS = [
    r"(?:可以\s*)?(?:帮我(?:记录|记下|记一下)|记录一下|记一下|记到苹果健康里|记到apple health里|保存一下|存一下)(?:吗|吧)?$",
]
_EN_LEADING_PATTERNS = [
    r"^(?:please\s+)?(?:help me\s+)?(?:log|record|save)\b",
    r"^(?:i\s+)?(?:had|ate|drank)\b",
]
_EN_TRAILING_PATTERNS = [
    r"\b(?:please\s+)?(?:log|record|save)(?:\s+(?:this|it|this meal))?$",
    r"\b(?:to\s+apple\s+health)$",
]
_CONNECTOR_PATTERNS = [
    r"^(?:就是|是|吃的是|大概是|应该是)",
    r"^(?:just|it was|probably|maybe)\b",
]

_DIRECT_TRANSLATIONS: list[tuple[str, str]] = [
    ("番茄肉酱意大利面", "spaghetti bolognese"),
    ("肉酱意大利面", "spaghetti bolognese"),
    ("意大利面", "spaghetti"),
    ("牛油果吐司", "avocado toast"),
    ("鸡胸肉", "chicken breast"),
    ("鸡蛋", "egg"),
    ("白米饭", "white rice"),
    ("米饭", "rice"),
    ("炒饭", "fried rice"),
    ("炒面", "fried noodles"),
    ("面条", "noodles"),
    ("馄饨", "wonton"),
    ("水饺", "dumplings"),
    ("饺子", "dumplings"),
    ("小笼包", "soup dumplings"),
    ("包子", "steamed bun"),
    ("豆腐", "tofu"),
    ("牛肉", "beef"),
    ("鸡肉", "chicken"),
    ("猪肉", "pork"),
    ("鱼", "fish"),
    ("虾", "shrimp"),
    ("沙拉", "salad"),
    ("酸奶", "yogurt"),
    ("燕麦", "oatmeal"),
]

_PUNCT_RE = re.compile(r"[，。！？、,.!?;:]+")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedMealQuery:
    user_text: str
    meal_description: str
    food_query: str
    food_query_en: str
    detected_language: str


def normalize_meal_query(
    *,
    user_text: str,
    meal_description: str | None = None,
    food_query: str | None = None,
    food_query_en: str | None = None,
) -> NormalizedMealQuery:
    raw_user_text = _normalize_spaces(user_text)
    meal_text = _normalize_spaces(meal_description or raw_user_text)
    normalized_food_query = _normalize_spaces(food_query or _extract_food_query(meal_text))
    if not normalized_food_query:
        normalized_food_query = meal_text

    detected_language = _detect_language(normalized_food_query or meal_text or raw_user_text)
    normalized_food_query_en = _normalize_english_query(food_query_en or _build_english_query(normalized_food_query, detected_language))
    return NormalizedMealQuery(
        user_text=raw_user_text,
        meal_description=meal_text,
        food_query=normalized_food_query,
        food_query_en=normalized_food_query_en,
        detected_language=detected_language,
    )


def _extract_food_query(text: str) -> str:
    candidate = _normalize_spaces(text)
    if not candidate:
        return ""

    candidate = _PUNCT_RE.sub(" ", candidate)
    for pattern in _ZH_LEADING_PATTERNS + _EN_LEADING_PATTERNS + _CONNECTOR_PATTERNS:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
    for pattern in _ZH_TRAILING_PATTERNS + _EN_TRAILING_PATTERNS:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()

    candidate = re.sub(r"\b(?:for\s+)?(?:breakfast|lunch|dinner|snack|today|tonight)\b", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"(?:今天|今晚|中午|早上|早餐|午餐|晚餐|夜宵)$", "", candidate).strip()
    candidate = re.sub(r"^(?:的|了|呢|呀)+", "", candidate).strip()
    candidate = re.sub(r"(?:吗|呢|呀|吧)+$", "", candidate).strip()
    return _normalize_spaces(candidate)


def _build_english_query(food_query: str, detected_language: str) -> str:
    normalized = _normalize_spaces(food_query)
    if not normalized:
        return ""
    if detected_language != "zh":
        return normalized

    query_en = normalized
    for zh, en in _DIRECT_TRANSLATIONS:
        query_en = query_en.replace(zh, en)

    if re.search(r"[\u4e00-\u9fff]", query_en):
        return normalized

    query_en = query_en.replace("和", " ")
    query_en = query_en.replace("配", " ")
    query_en = query_en.replace("加", " ")
    query_en = _PUNCT_RE.sub(" ", query_en)
    query_en = re.sub(r"[\u4e00-\u9fff]+", " ", query_en)
    query_en = _normalize_english_query(query_en)
    return query_en or normalized


def _detect_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _normalize_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").strip())


def _normalize_english_query(text: str) -> str:
    return _normalize_spaces(text).lower()
