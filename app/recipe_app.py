import logging
import os
from pathlib import Path
import time
from typing import Any, Dict
from uuid import uuid4

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage  # noqa: F401
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from app.observability import is_terminal_trace_enabled, reset_trace_id, set_trace_id, trace_log
from app.observability.agent_middleware import AgentContextTraceMiddleware
from app.tools.tool_registry import load_all_tools

load_dotenv()

SYSTEM_PROMPT = (
    "You are a professional chef knowledgeable in both Chinese and Western cuisine, "
    "with strong cooking experience and nutrition expertise. "
    "Your job is to recommend suitable recipes or practical cooking advice based on the user's "
    "diet preferences, taste, available ingredients, and health needs. "
    "If the user is trying to lose fat, recommend low-calorie, high-protein, light dishes. "
    "If the user wants to improve everyday family meals, suggest easy-to-cook, nutritious home-style dishes. "
    "If the user mentions specific ingredients (e.g., chicken, tomatoes), recommend appropriate techniques "
    "based on the ingredient characteristics. "
    "\n\n"
    "Use memory files for stable user preferences and profile:\n"
    "- Long-term memory directory: /memories/users/<user_id>/\n"
    "- Phase 1 core memory files are /memories/users/<user_id>/profile.md and /memories/users/<user_id>/preferences.md\n"
    "- profile.md is for stable user background such as confirmed allergies, intolerances, or long-term dietary constraints\n"
    "- preferences.md is for stable food, cuisine, budget, and cooking preferences\n"
    "- goals.md and shopping.md may exist later, but do not overuse them in the current phase\n"
    "Read existing memory before giving personalized advice. "
    "Only update long-term memory when the user provides durable, explicit information.\n"
    "- Safe auto-write categories: explicit allergies or intolerances, explicit long-term diet goals, explicit stable preferences, and explicit stable shopping habits\n"
    "- Do not write one-off requests, temporary moods, meal logs, weight history, Apple Health sync state, or packaged-food catalog records into long-term memory\n"
    "- Do not write inferred medical conclusions, sensitive identity details, or guesses\n"
    "- If information is ambiguous or only relevant to the current chat, keep it in conversation context and do not write memory\n"
    "- When updating memory, prefer revising existing stable facts instead of appending noisy event history\n\n"
    "Meal logging policy:\n"
    "- If the user asks to record/log a meal, first determine whether you already have a usable final nutrition estimate.\n"
    "- If the user has already asked to log or record a meal and the food is identifiable, do not ask an extra "
    "permission question about creating a draft. If you already have a usable final estimate, prepare the draft immediately, then show the estimate and ask only for save confirmation.\n"
    "- Before calling prepare_meal_log, first use one or more nutrition lookup tools and choose a single final estimate for kcal, protein, carbs, and fat.\n"
    "- Choose nutrition tools based on food type and source fit. Do not assume a fixed tool priority order.\n"
    "- For Chinese or mixed-language foods, you may derive an English lookup query for tools that work better in English.\n"
    "- Pass meal_description plus the final nutrition estimate and source into prepare_meal_log. prepare_meal_log does not perform lookup for you.\n"
    "- If you do not have a usable final estimate, do not call prepare_meal_log yet. Ask one concise clarification question or explicitly choose an approximate estimate first.\n"
    "- Never write meal logs without explicit confirmation.\n"
    "- Only after explicit confirmation, use tool commit_meal_log.\n"
    "- If confirmation is missing, ask for confirmation first."
)

SKILLS_ROUTE = "/skills/"


class RecipeApp:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing environment variable OPENAI_API_KEY. Please configure it in your .env file.")

        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=api_key,
            temperature=0.5,
        )
        self.memory_saver = MemorySaver()
        self.tools = load_all_tools()
        self.deep_agent = self._build_deep_agent()

    @staticmethod
    def _history_key(user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    @staticmethod
    def _skills_root() -> Path:
        return Path(__file__).resolve().parent / "skills"

    @staticmethod
    def _build_backend(runtime: Any, *, skills_root: Path | None = None) -> CompositeBackend:
        resolved_skills_root = (skills_root or RecipeApp._skills_root()).resolve()
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/memories/": StoreBackend(runtime),
                SKILLS_ROUTE: FilesystemBackend(root_dir=resolved_skills_root, virtual_mode=True),
            },
        )

    def _build_deep_agent(self):
        composite_backend = lambda runtime: self._build_backend(runtime, skills_root=self._skills_root())
        middleware = [AgentContextTraceMiddleware()] if is_terminal_trace_enabled() else []

        return create_deep_agent(
            model=self.model,
            tools=self.tools,
            skills=[SKILLS_ROUTE],
            system_prompt=SYSTEM_PROMPT,
            middleware=middleware,
            backend=composite_backend,
            store=InMemoryStore(),
            checkpointer=self.memory_saver,
        )

    async def chat(self, chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        logging.info("[Chat] user_id=%s, chat_id=%s, message=%s", user_id, chat_id, message)
        trace_id = uuid4().hex[:12]
        trace_token = set_trace_id(trace_id)

        history_key = self._history_key(user_id, chat_id)
        config = {"configurable": {"thread_id": history_key}}
        runtime_instructions = (
            f"Current user_id: {user_id}\n"
            f"Current chat_id: {chat_id}\n"
            "Memory file paths for this user:\n"
            f"- /memories/users/{user_id}/profile.md\n"
            f"- /memories/users/{user_id}/preferences.md\n"
        )
        state: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": runtime_instructions},
                {"role": "user", "content": message},
            ]
        }
        trace_log(
            "ChatInput",
            "Invoking deep agent",
            {
                "user_id": user_id,
                "chat_id": chat_id,
                "thread_id": history_key,
                "messages": state["messages"],
            },
        )

        try:
            start = time.monotonic()
            out = await self.deep_agent.ainvoke(state, config=config)
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            answer = self._extract_answer(out)
            logging.info("[Chat] response=%s", answer)
            trace_log(
                "ChatOutput",
                "Deep agent returned",
                {
                    "elapsed_ms": elapsed_ms,
                    "answer": answer,
                    "result_keys": list(out.keys()) if isinstance(out, dict) else str(type(out)),
                },
            )
            return answer
        except Exception:
            logging.exception("[Chat] Invocation failed.")
            trace_log("ChatError", "Deep agent invocation failed")
            raise
        finally:
            reset_trace_id(trace_token)

    async def close(self) -> None:
        return None

    @staticmethod
    def _extract_answer(result: Dict[str, Any]) -> str:
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict):
                if msg.get("role") != "assistant":
                    continue
                return RecipeApp._normalize_content(msg.get("content", ""))

            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role not in ("assistant", "ai"):
                continue
            return RecipeApp._normalize_content(getattr(msg, "content", ""))
        return ""

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        chunks.append(text)
                elif isinstance(part, str):
                    chunks.append(part)
            return "\n".join(chunks).strip()
        return str(content or "")
