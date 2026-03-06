# AI Diet Assistant (Deep Agent Baseline)

This repository is currently in a **baseline architecture phase**.

The app runs a **minimal Deep Agent pipeline** behind FastAPI, with no RAG and no external tools yet.  
Goal: stabilize the core chat, memory, and runtime interfaces first, then add capabilities incrementally.

## Current Status

### What is enabled
- Deep Agents-based chat pipeline
- FastAPI Web UI + API endpoints
- Session memory via LangGraph checkpointer
- In-memory long-term memory filesystem route (`/memories/`)
- AgentCore-compatible endpoints (`/ping`, `/invocations`)

### What is intentionally disabled (for now)
- RAG retrieval pipeline
- Tool registry / MCP tools
- Report generation flow
- Postgres-backed persistent memory

## Architecture (Current)

### 1) Runtime Layer
- `main.py` exposes:
  - `GET /` web chat UI
  - `POST /chat` local chat API
  - `GET /ping` health check
  - `POST /invocations` AgentCore runtime entrypoint

### 2) Agent Layer
- `app/recipe_app.py` builds a minimal Deep Agent:
  - `create_deep_agent(...)`
  - `tools=[]` (empty by design in baseline)
  - chef-oriented `system_prompt`

### 3) Memory Layer
- `MemorySaver()` for thread/session state
- `InMemoryStore()` for store-backed paths
- `CompositeBackend` routing:
  - default: `StateBackend`
  - `/memories/`: `StoreBackend`

This gives:
- short-term memory in-thread
- long-term memory semantics through `/memories/` (currently in-memory only)

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

### Phase 2: Channel Integration (SMS first)
- Connect iPhone SMS user flow through a messaging gateway webhook
- Let users chat with the diet assistant through SMS

### Phase 3: Meal Photo Nutrition
- Accept user meal photos
- Estimate calories and macro nutrients
- Return structured nutrition summary with confidence

### Phase 4: Apple Health App Sync
- Add iOS companion app for HealthKit authorization/write
- Sync confirmed nutrition entries into Apple Health
- Extend to additional health/productivity apps later

### Phase 5: Skills-based Tooling
- Replace flat tools with Skills-managed capability packs
- Progressive loading of skills by task
- Add governance/audit and high-risk action controls

## Notes

- Current memory store is **InMemoryStore** for development speed.
- Persistent store (e.g., Postgres-backed store) will be added later after baseline stabilization.
- Tests that depend on older RAG/tools pipeline will be migrated in subsequent iterations.
