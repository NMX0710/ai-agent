import sys
from pathlib import Path

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))


def test_search_youtube_playlist_recipes_returns_ranked_results(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setenv("YOUTUBE_PLAYLIST_ID", "playlist-123")
    monkeypatch.setenv("YOUTUBE_PLAYLIST_MAX_ITEMS", "100")

    import app.settings as settings
    import app.tools.youtube_playlist_tools as youtube_tools

    monkeypatch.setattr(settings, "YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setattr(settings, "YOUTUBE_PLAYLIST_ID", "playlist-123")
    monkeypatch.setattr(settings, "YOUTUBE_PLAYLIST_MAX_ITEMS", 100)
    monkeypatch.setattr(settings, "YOUTUBE_SUMMARY_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_ID", "playlist-123")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_MAX_ITEMS", 100)
    monkeypatch.setattr(youtube_tools, "_enrich_video_row", lambda row: {**row, "summary": "High protein dinner bowl with chicken and rice.", "summary_source": "test", "transcript_available": True})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            if "playlistItems" in url:
                return _FakeResponse(
                    {
                        "items": [
                            {
                                "snippet": {
                                    "title": "High Protein Chicken Rice Bowl",
                                    "description": "Easy high protein dinner bowl for weeknight meal prep.",
                                    "publishedAt": "2026-03-20T00:00:00Z",
                                    "channelTitle": "Test Channel",
                                    "resourceId": {"videoId": "video-1"},
                                }
                            },
                            {
                                "snippet": {
                                    "title": "Chocolate Overnight Oats",
                                    "description": "Breakfast prep idea.",
                                    "publishedAt": "2026-03-10T00:00:00Z",
                                    "channelTitle": "Test Channel",
                                    "resourceId": {"videoId": "video-2"},
                                }
                            },
                        ]
                    }
                )
            return _FakeResponse({"items": []})

    monkeypatch.setattr("app.tools.youtube_playlist_tools.httpx.Client", _FakeClient)

    result = youtube_tools.search_youtube_playlist_recipes.func(query="high protein dinner", max_results=2)

    assert result["count"] == 1
    assert result["source"] == "youtube_playlist_with_search_fallback"
    assert result["results"][0]["title"] == "High Protein Chicken Rice Bowl"
    assert result["results"][0]["url"] == "https://www.youtube.com/watch?v=video-1"
    assert result["results"][0]["match_score"] > 0
    assert result["results"][0]["source_kind"] == "playlist"
    assert result["used_general_search_fallback"] is False
    assert result["results"][0]["summary"] == "High protein dinner bowl with chicken and rice."


def test_search_youtube_playlist_recipes_handles_missing_configuration(monkeypatch):
    import app.tools.youtube_playlist_tools as youtube_tools

    monkeypatch.setattr(youtube_tools, "YOUTUBE_API_KEY", "")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_ID", "")

    result = youtube_tools.search_youtube_playlist_recipes.func(query="chicken meal prep")

    assert result == {"error": "missing YOUTUBE_API_KEY"}


def test_search_youtube_playlist_recipes_returns_empty_when_no_match(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setenv("YOUTUBE_PLAYLIST_ID", "playlist-123")

    import app.tools.youtube_playlist_tools as youtube_tools

    monkeypatch.setattr(youtube_tools, "YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_ID", "playlist-123")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_MAX_ITEMS", 50)
    monkeypatch.setattr(youtube_tools, "_enrich_video_row", lambda row: {**row, "summary": "", "summary_source": "test", "transcript_available": False})

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            if "playlistItems" in url:
                return _FakeResponse(
                    {
                        "items": [
                            {
                                "snippet": {
                                    "title": "Chocolate Cake Tutorial",
                                    "description": "Birthday cake decoration ideas.",
                                    "publishedAt": "2026-03-20T00:00:00Z",
                                    "channelTitle": "Test Channel",
                                    "resourceId": {"videoId": "video-3"},
                                }
                            }
                        ]
                    }
                )
            return _FakeResponse({"items": []})

    monkeypatch.setattr("app.tools.youtube_playlist_tools.httpx.Client", _FakeClient)

    result = youtube_tools.search_youtube_playlist_recipes.func(query="high protein dinner", max_results=3)

    assert result["count"] == 0
    assert result["results"] == []


def test_search_youtube_playlist_recipes_falls_back_to_general_search(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setenv("YOUTUBE_PLAYLIST_ID", "playlist-123")

    import app.tools.youtube_playlist_tools as youtube_tools

    monkeypatch.setattr(youtube_tools, "YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_ID", "playlist-123")
    monkeypatch.setattr(youtube_tools, "YOUTUBE_PLAYLIST_MAX_ITEMS", 50)
    monkeypatch.setattr(
        youtube_tools,
        "_enrich_video_row",
        lambda row: {
            **row,
            "summary": "Protein-packed dinner prep." if row["video_id"] == "video-4" else "",
            "summary_source": "test",
            "transcript_available": row["video_id"] == "video-4",
        },
    )

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            if "playlistItems" in url:
                return _FakeResponse(
                    {
                        "items": [
                            {
                                "snippet": {
                                    "title": "Chocolate Cake Tutorial",
                                    "description": "Birthday cake decoration ideas.",
                                    "publishedAt": "2026-03-20T00:00:00Z",
                                    "channelTitle": "Test Channel",
                                    "resourceId": {"videoId": "video-3"},
                                }
                            }
                        ]
                    }
                )
            return _FakeResponse(
                {
                    "items": [
                        {
                            "id": {"videoId": "video-4"},
                            "snippet": {
                                "title": "High Protein Dinner Meal Prep",
                                "description": "Easy dinner prep with lots of protein.",
                                "publishedAt": "2026-03-21T00:00:00Z",
                                "channelTitle": "Fallback Channel",
                            },
                        }
                    ]
                }
            )

    monkeypatch.setattr("app.tools.youtube_playlist_tools.httpx.Client", _FakeClient)

    result = youtube_tools.search_youtube_playlist_recipes.func(query="high protein dinner", max_results=3)

    assert result["count"] == 1
    assert result["playlist_count"] == 0
    assert result["used_general_search_fallback"] is True
    assert result["results"][0]["source_kind"] == "youtube_search"
    assert result["results"][0]["url"] == "https://www.youtube.com/watch?v=video-4"


def test_enrich_video_row_uses_cache_before_transcript_fetch(monkeypatch, tmp_path):
    import app.youtube.store as youtube_store
    import app.tools.youtube_playlist_tools as youtube_tools

    monkeypatch.setattr(youtube_store, "YOUTUBE_CACHE_DIR", tmp_path)
    youtube_store.save_video_cache(
        "video-cache-1",
        {
            "video_id": "video-cache-1",
            "summary": "Cached chicken meal prep summary.",
            "summary_source": "cache",
            "transcript_available": True,
        },
    )

    called = {"transcript": False, "summary": False}

    def _fail_transcript(video_id):
        called["transcript"] = True
        raise AssertionError("transcript fetch should not run on cache hit")

    def _fail_summary(**kwargs):
        called["summary"] = True
        raise AssertionError("summary generation should not run on cache hit")

    monkeypatch.setattr(youtube_tools, "load_video_cache", youtube_store.load_video_cache)
    monkeypatch.setattr(youtube_tools, "save_video_cache", youtube_store.save_video_cache)
    monkeypatch.setattr(youtube_tools, "fetch_video_transcript", _fail_transcript)
    monkeypatch.setattr(youtube_tools, "summarize_recipe_video", _fail_summary)

    row = {
        "video_id": "video-cache-1",
        "title": "Meal Prep Video",
        "description": "desc",
        "url": "https://youtube.com/watch?v=video-cache-1",
        "source_kind": "playlist",
        "match_score": 10,
    }
    result = youtube_tools._enrich_video_row(row)

    assert result["summary"] == "Cached chicken meal prep summary."
    assert result["summary_source"] == "cache"
    assert result["transcript_available"] is True
    assert called == {"transcript": False, "summary": False}
