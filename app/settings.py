import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# root dir
PROJECT_ROOT = Path(__file__).parent.parent
FILE_SAVE_DIR = PROJECT_ROOT / "tmp" / "files"

# Long-term memory settings
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ai_agent:ai_agent_dev@localhost:5432/ai_agent",
)
LONG_TERM_MEMORY_ENABLED = os.getenv("LONG_TERM_MEMORY_ENABLED", "1") == "1"
MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "90"))
MEMORY_MAX_RECORDS_PER_USER = int(os.getenv("MEMORY_MAX_RECORDS_PER_USER", "200"))
MEMORY_RETRIEVE_TOP_K = int(os.getenv("MEMORY_RETRIEVE_TOP_K", "5"))

# Telegram settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWLIST_RAW = os.getenv("TELEGRAM_ALLOWLIST", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

# YouTube playlist retrieval settings
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_PLAYLIST_ID = os.getenv("YOUTUBE_PLAYLIST_ID", "")
YOUTUBE_PLAYLIST_MAX_ITEMS = int(os.getenv("YOUTUBE_PLAYLIST_MAX_ITEMS", "100"))
YOUTUBE_SUMMARY_MODEL = os.getenv("YOUTUBE_SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
YOUTUBE_CACHE_DIR = Path(os.getenv("YOUTUBE_CACHE_DIR", str(PROJECT_ROOT / "tmp" / "youtube_cache")))

# Apple Health bridge settings
APPLE_HEALTH_BRIDGE_TOKEN = os.getenv("APPLE_HEALTH_BRIDGE_TOKEN", "")
