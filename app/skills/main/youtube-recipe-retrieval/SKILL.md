---
name: youtube-recipe-retrieval
description: Use this skill when users want meal ideas, recipe inspiration, or weekly meal-prep suggestions from the configured YouTube playlist.
---

# YouTube Recipe Retrieval

## Overview

Use this skill when the user wants help deciding what to eat, asks for recipe ideas, or wants a weekly meal-prep plan with a video reference.

## When To Use

Use `search_youtube_playlist_recipes(...)` for requests like:
- "What should I eat tonight?"
- "Give me a chicken breast recipe."
- "Help me plan a week of meal prep."
- "Show me a video for a high-protein dinner."

## Tool Use Rules

1. Build a concise English recipe-style search query before calling the tool.
- Prefer dish names, meal types, or short need phrases like `high protein dinner`, `chicken rice bowl`, or `weekly meal prep`.
- Do not pass long conversational sentences if a shorter recipe query is clearer.

2. Use this tool for recipe inspiration, not nutrition estimation.
- Do not treat playlist videos as a precise calorie or macro source.
- If the user is primarily asking for calories, protein, carbs, or fat, use the nutrition-specialist flow instead.

3. Prefer one strong video reference over many weak ones.
- Usually return one primary video link in the final answer.
- You may mention one or two additional ideas when helpful, but keep the response concise.
- If the tool returns a `summary`, use it to explain why the video fits the user's need before giving the link.

4. Include the direct YouTube URL in the final answer.
- Telegram can render a link preview from the standard YouTube URL.
- Keep the link intact as a raw standard YouTube URL on its own line.
- Do not wrap the URL in markdown, angle brackets, or surrounding punctuation.

5. If the playlist search returns no relevant result:
- The tool may fall back to general YouTube video search.
- Prefer playlist videos when available.
- If you use a fallback result, say it is an additional YouTube reference rather than pretending it came from the playlist.

## Answer Style

- Give a practical food recommendation first.
- If available, briefly summarize what the chosen video covers.
- Then attach the most relevant video link as a reference.
- Respond in the user's language when practical.
