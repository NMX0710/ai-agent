# AGENTS.md

## Project Overview
- This repository is an AI diet assistant built around a FastAPI app and a Deep Agents runtime.
- `main.py` is the runtime entrypoint. It serves the web UI, local chat API, Telegram webhook, health endpoints, and Apple Health bridge endpoints.
- `app/recipe_app.py` builds the active agent pipeline: OpenAI chat model, Deep Agent orchestration, tool loading, and memory routing.
- `app/tools/` contains tool entrypoints. `app/nutrition/` contains meal logging, nutrition estimation, Apple Health mapping, and draft state.
- `app/governance/` handles image-event processing. `app/memory/` contains long-term memory helpers. `app/skills/` holds skill-specific instructions for the agent runtime.
- `tests/` is the source of truth for expected behavior. Some tests and modules still reflect older RAG-era architecture; treat them as legacy unless your task is explicitly reviving that path.

## Agent Roles

### Feature Agent
- Implement one user-visible behavior at a time.
- Prefer extending existing flows over introducing new abstractions.
- Follow the current runtime shape: FastAPI endpoint -> app/service layer -> tools/domain objects -> tests.

### Refactor Agent
- Improve clarity, naming, structure, or boundaries without changing behavior.
- Keep refactors local. If the blast radius expands beyond the original target, stop and split the work.
- Do not rewrite active flows to match stale tests or deprecated architecture.

### Test Agent
- Add or update tests around the changed behavior.
- Prefer focused pytest coverage near the touched module.
- Use mocks/monkeypatching for external APIs, LLM calls, Telegram, and Apple Health bridge behavior.

### Documentation Agent
- Update docs only when behavior, setup, architecture, or workflows materially change.
- Keep docs operational and brief. Document what to run, what changed, and what remains intentionally disabled.

## Atomic Commit Rules
- Keep changes small and reversible.
- Only stage files you edited. Never use `git add .` or broad staging patterns.
- Do not revert unrelated changes. Other agents may be working in the same tree.
- Check `git diff` before staging and commit only the intended hunk set.
- If a task naturally splits into separate concerns, make separate commits.
- Prefer one commit per coherent behavior change, test addition, or refactor.

## Development Guidelines
- Read the relevant code first. Start from the entrypoint and follow the real call path before editing.
- Think in blast radius: estimate how many files, tests, and workflows a change will touch.
- Prefer small direct changes over framework-building or speculative abstractions.
- Preserve the current baseline architecture unless the task explicitly changes it.
- This is an agent-driven system built on Deep Agents. Prefer improving agent instructions, tool contracts, skills, and native agent planning before adding regex-heavy or rule-heavy logic.
- Use rule-based parsing only as a thin fallback or safety net. Do not turn core product behavior into a rules engine when the agent can make the decision natively.
- Keep environment-driven behavior in `app/settings.py` or existing config points.
- Reuse existing modules and patterns:
  - FastAPI routes in `main.py` or router modules.
  - Tool registration in `app/tools/tool_registry.py`.
  - Domain logic in `app/nutrition/`, `app/governance/`, or `app/memory/`.
- Treat external integrations as unstable boundaries. Fail safely and log clearly.

## Task Workflow
1. Read the relevant files and tests first.
2. Estimate blast radius before editing.
3. If the change is large, propose smaller options or phased slices before touching many files.
4. Implement the smallest useful slice.
5. Run targeted verification.
6. Review diff for unrelated fallout before staging.

## Testing Rules
- Run the narrowest pytest target that proves the change.
- Add tests when changing behavior, fixing a bug, or touching parsing, webhook, memory, meal-log, or tool logic.
- Prefer deterministic tests. Mock network, model, and third-party API calls.
- If a legacy test suite is already broken for unrelated reasons, do not broaden the repair unless the task requires it. Note the gap explicitly.
- For endpoint changes, prefer `fastapi.testclient` coverage.

## Documentation Rules
- Update `README.md` when setup, endpoints, env vars, or user-facing workflows change.
- Update skill docs when a task changes behavior inside `app/skills/`.
- Do not create documentation for speculative designs.
- Keep documentation aligned with the active baseline architecture, not retired RAG assumptions.

## Safety Rules
- Do not modify files outside the task scope.
- Assume concurrent agents may edit nearby files. Re-read before patching if a file changed during your work.
- Avoid destructive git operations unless explicitly requested.
- Do not silently change API contracts, env var names, persistence semantics, or webhook behavior.
- Never enable external side effects by default in tests.
- If a change touches multiple subsystems or public contracts, prefer an iterative rollout over a single large patch.

## Common Workflows
- Install deps: `pip install -r requirements.txt`
- Run app: `uvicorn main:app --reload`
- Run tests: `pytest`, or targeted paths such as `pytest tests/test_telegram_webhook.py`
- Start local Postgres when needed: `docker compose up postgres`
