# AI Diet Assistant (Deep Agent Baseline)

This repository is currently in a **baseline architecture phase**.

The app runs a **minimal Deep Agent pipeline** behind FastAPI, with no RAG.  
Goal: stabilize the core chat, memory, and runtime interfaces first, then add capabilities incrementally.

## Current Status

### What is enabled
- Deep Agents-based chat pipeline
- FastAPI Web UI + API endpoints
- Session memory via LangGraph checkpointer
- In-memory long-term memory filesystem route (`/memories/`)
- Telegram webhook channel (`/webhooks/telegram`) with allowlist + update dedup
- Built-in HTTP nutrition tools for USDA, Spoonacular, Tavily fallback, and Open Food Facts
- Meal logging draft -> confirm -> Apple Health bridge flow
- Agent-side nutrition tool selection with source-labeled final estimates
- Conservative long-term memory policy for `/memories/users/<user_id>/*.md`
- Local Apple Health bridge runner CLI with mock writers for end-to-end bridge verification
- AgentCore-compatible endpoints (`/ping`, `/invocations`)

### What is intentionally disabled (for now)
- RAG retrieval pipeline
- Report generation flow
- Postgres-backed persistent memory

## Architecture (Current)

### 1) Runtime Layer
- `main.py` exposes:
  - `GET /` web chat UI
  - `POST /chat` local chat API
  - `POST /webhooks/telegram` Telegram inbound webhook
  - `GET /ping` health check
  - `POST /invocations` AgentCore runtime entrypoint
  - `POST /integrations/apple-health/pending-writes`
  - `POST /integrations/apple-health/write-result`

### 2) Agent Layer
- `app/recipe_app.py` builds a minimal Deep Agent:
  - `create_deep_agent(...)`
  - nutrition tools loaded at startup (configured by env)
  - nutrition and memory policy skills loaded from project filesystem via `/skills/`
  - chef-oriented `system_prompt`
  - explicit meal-log guidance for choosing a nutrition source before draft creation

### 3) Memory Layer
- `MemorySaver()` for thread/session state
- `InMemoryStore()` for store-backed paths
- `CompositeBackend` routing:
  - default: `StateBackend`
  - `/memories/`: `StoreBackend`
  - `/skills/`: `FilesystemBackend` rooted at `app/skills`

This gives:
- short-term memory in-thread
- long-term memory semantics through `/memories/` (currently in-memory only)
- Phase 1 long-term memory focus on `profile.md` and `preferences.md`

### 4) Meal Logging + Apple Health Bridge
- The agent chooses nutrition lookup tools and selects one final estimate before any write.
- `prepare_meal_log` creates a draft from that final estimate; it does not perform nutrition lookup itself.
- Telegram confirm/cancel buttons control the explicit write gate.
- Confirmed drafts become bridge-visible pending writes.
- `app/apple_health_bridge_runner.py` can poll pending writes and report results back.
- Current runner modes are `mock-success` and `mock-failure` for local end-to-end verification.
- Real Apple Health writes still require an Apple-platform companion implementation using HealthKit.

## Project Structure

```text
app/
├── recipe_app.py              # Minimal Deep Agent pipeline
├── templates/
│   └── index.html             # Web chat UI
├── static/
│   └── css/
└── routers/
    └── sample.py

main.py                        # FastAPI entrypoint
requirements.txt
tests/
```

## Local Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Configure environment
Create/update `.env` in project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini

# Spoonacular tool
SPOONACULAR_API_KEY=your_spoonacular_key

# USDA FoodData Central tool
USDA_API_KEY=your_usda_key

# Tavily fallback tool
TAVILY_API_KEY=your_tavily_key

# Open Food Facts identifies clients via User-Agent; no API key required for read access
OPENFOODFACTS_USER_AGENT="ai-agent-nutrition/0.1 (contact: team@example.com)"

# Telegram channel
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWLIST=123456789
TELEGRAM_WEBHOOK_SECRET=your_secret_token

# Optional Apple Health bridge auth
APPLE_HEALTH_BRIDGE_TOKEN=shared_bridge_secret
```

### 3) Run locally
```bash
uvicorn main:app --reload
```

After startup:
- Web UI: `http://127.0.0.1:8000`
- Chat API: `POST /chat`

`/chat` request body:
- `user_id`: stable user id
- `chat_id`: conversation id
- `message`: user input

Example:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id":"manual-chat-1",
    "user_id":"manual-user-1",
    "message":"Give me a high-protein low-fat dinner idea."
  }'
```

## Product Roadmap (Planned)

### Phase 1: Core Assistant (now)
- Keep architecture minimal and stable
- Validate chat quality and memory behavior
- Keep API contracts fixed

### Phase 2: Channel Integration (Telegram first)
- Use Telegram as the primary mobile chat channel
- Keep iterating the Telegram conversation quality and reliability

### Phase 3: Meal Photo Nutrition
- Accept user meal photos
- Estimate calories and macro nutrients
- Return structured nutrition summary with confidence

### Phase 4: Apple Health App Sync
- Add iOS companion app for HealthKit authorization/write
- Replace mock bridge writer with real HealthKit writer
- Sync confirmed nutrition entries into Apple Health
- Extend to additional health/productivity apps later

### Phase 5: Skills-based Tooling
- Replace flat tools with Skills-managed capability packs
- Progressive loading of skills by task
- Add governance/audit and high-risk action controls

## Notes

- Current memory store is **InMemoryStore** for development speed.
- Persistent store (e.g., Postgres-backed store) will be added later after baseline stabilization.
- Long-term memory policy is documented in `docs/long-term-memory-design.md`.
- Telegram allowlist uses numeric user IDs (`from.id`) for stability.
- Current media policy is conservative: image logging flow is not enabled yet; text chat is the current stable path.
- Nutrition tool selection is guided by skill/tool descriptions rather than a hard-coded source priority list in the meal-log service.
- Nutrition source labels are kept explicit: `USDA`, `Spoonacular`, `Tavily`, `OpenFoodFacts`, `Estimated`.
- Open Food Facts read access does not require an API key, but requests should send a descriptive `User-Agent`.
- The Python bridge runner verifies the server-side Apple Health queue lifecycle, but it does not write to HealthKit by itself.
- Tests that depend on older RAG/tools pipeline will be migrated in subsequent iterations.

## Apple Health Bridge Runner

Run the API server first, then start the bridge runner in a separate process:

```bash
./.venv/bin/python -m app.apple_health_bridge_runner \
  --base-url http://127.0.0.1:8000 \
  --user-id tg:123 \
  --writer mock-success \
  --lease-seconds 120 \
  --once
```

Supported runner modes:
- `mock-success`: consumes a pending write and reports a successful sync
- `mock-failure`: consumes a pending write and reports a failed sync

This is intended for bridge verification only. A real Apple Health write path still requires a native iOS/macOS writer backed by HealthKit.

## Telegram Quick Test

1. Start app:
```bash
uvicorn main:app --reload
```

2. Expose local service (example):
```bash
ngrok http 8000
```

3. Set Telegram webhook:
```bash
curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://<your-public-domain>/webhooks/telegram",
    "secret_token": "your_secret_token"
  }'
```

4. Send a message to your bot from your phone Telegram app and verify reply.
