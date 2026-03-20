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

from app.observability.backend_trace import TracingBackend
from app.observability import is_terminal_trace_enabled, reset_trace_id, set_trace_id, trace_log
from app.observability.agent_middleware import AgentContextTraceMiddleware
from app.tools.tool_registry import load_all_tools

load_dotenv()

SYSTEM_PROMPT = (
    "You are a diet and cooking assistant with strong Chinese and Western cooking knowledge and practical nutrition sense. "
    "Help users with recipe advice, meal ideas, cooking guidance, and nutrition questions in a clear, practical way. "
    "Respond in the user's language when practical. "
    "Do not pretend to know details you do not have; ask a brief clarification question when needed. "
    "\n\n"
    "Your long-term memory lives under /memories/users/<user_id>/. "
    "Use /memories/users/<user_id>/profile.md for durable dietary constraints, allergies, intolerances, and other stable user facts that materially affect recommendations. "
    "Use /memories/users/<user_id>/preferences.md for stable food likes or dislikes, recurring eating habits, cuisine preferences, ingredient avoidances, and stable budget or cooking-time preferences. "
    "For recurring eating habits and dietary preferences, write only to the canonical file /memories/users/<current user_id>/preferences.md. "
    "Always use the current runtime user_id when writing long-term memory. Never substitute placeholder ids such as default_user or anonymous. "
    "Do not invent new memory filenames such as dietary_preferences.md, dietary_habits.md, or other free-form variants for preference memory. "
    "Write preference memory as short bullet summaries in preferences.md rather than loose paragraphs. "
    "When the user explicitly states a stable preference or recurring habit with strong wording such as usually, normally, always, do not eat, avoid, 我平常, 我一般, or 我不吃, do not ask a redundant confirmation question before updating long-term memory unless the user also indicates uncertainty or that it is temporary. "
    "If you update long-term memory, update the canonical target file directly. "
    "When a user explicitly states a durable fact or stable preference that is likely to matter again, consider updating long-term memory even if the user does not explicitly ask you to remember it, following the memory-policy skill. "
    "Do not write temporary requests, one-time moods, meal events, weight history, Apple Health sync state, or inferred medical conclusions into long-term memory. "
    "Use the appropriate skills and tools for task-specific workflows instead of inventing ad hoc rules. "
    "For any request that requires calorie, macro, or nutrition estimation, delegate the estimation work to the nutrition-specialist subagent instead of answering from general knowledge. "
    "This includes direct nutrition questions and meal-log pre-estimate tasks. "
    "The nutrition-specialist only estimates nutrition; it does not prepare drafts, commit meal logs, or handle Apple Health flow. "
    "If the user wants to log a meal, first delegate the nutrition estimate to the nutrition-specialist, then handle prepare_meal_log and later confirmation in the main agent yourself. "
    "Do not use the general-purpose subagent for nutrition estimation when the nutrition-specialist fits the task. "
    "If the user is only asking a nutrition question, answer that question directly. "
    "Only enter meal logging flow when the user clearly asks to log, save, or record intake. "
    "Never commit a meal log without explicit user confirmation."
)

MAIN_SKILLS_ROUTE = "/skills/main/"
NUTRITION_SKILLS_ROUTE = "/skills/nutrition/"
NUTRITION_TOOL_NAMES = {
    "spoonacular_search_recipe",
    "usda_search_foods",
    "tavily_search_nutrition",
    "openfoodfacts_search_products",
}


class RecipeApp:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing environment variable OPENAI_API_KEY. Please configure it in your .env file.")

        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            api_key=api_key,
            temperature=0.5,
        )
        self.memory_saver = MemorySaver()
        self.tools = self._main_agent_tools()
        self.deep_agent = self._build_deep_agent()

    @staticmethod
    def _history_key(user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    @staticmethod
    def _skills_root() -> Path:
        return Path(__file__).resolve().parent / "skills"

    @staticmethod
    def _nutrition_tools() -> list[Any]:
        return [
            tool
            for tool in load_all_tools()
            if getattr(tool, "name", None) in NUTRITION_TOOL_NAMES
        ]

    @staticmethod
    def _main_agent_tools() -> list[Any]:
        return [
            tool
            for tool in load_all_tools()
            if getattr(tool, "name", None) not in NUTRITION_TOOL_NAMES
        ]

    @staticmethod
    def _build_backend(runtime: Any, *, skills_root: Path | None = None) -> CompositeBackend:
        resolved_skills_root = (skills_root or RecipeApp._skills_root()).resolve()
        backend = CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/memories/": StoreBackend(runtime),
                "/skills/": FilesystemBackend(root_dir=resolved_skills_root, virtual_mode=True),
            },
        )
        return TracingBackend(backend)

    def _build_deep_agent(self):
        composite_backend = lambda runtime: self._build_backend(runtime, skills_root=self._skills_root())
        middleware = [AgentContextTraceMiddleware()] if is_terminal_trace_enabled() else []
        nutrition_specialist = {
            "name": "nutrition-specialist",
            "description": (
                "Use this agent for all calorie, macro, and nutrition estimation tasks. "
                "It handles ambiguous foods, provider selection, clarification questions, and choosing one final estimate. "
                "It does not perform meal-log actions."
            ),
            "system_prompt": (
                "You are a nutrition estimation specialist. "
                "Use the nutrition-lookup skill for workflow guidance. "
                "Use only the provided nutrition lookup tools. "
                "Your job is estimation only, not logging orchestration. "
                "Before returning a final estimate, verify that the candidate matches the user's portion level and meal context. "
                "Do not treat a likely per-100g value, a small-serving database value, or a generic low-granularity entry as the final estimate for a full dish or whole meal. "
                "When the user describes a full dish or meal and direct lookup results are missing, low-granularity, or implausible, decompose the dish into common ingredients and estimate from a typical single serving. "
                "Use any user-provided ingredients to override the default composition before estimating. "
                "If the food description is too ambiguous for a credible estimate, ask one concise, high-value clarification question. "
                "If portion size is missing, either ask one concise, high-value clarification question or estimate a common single serving and clearly label that assumption as approximate. "
                "If the user wants an estimate without more detail, choose a common default version of the dish and clearly label it as approximate. "
                "If an estimate is implausibly low or high for the described dish or meal, do not return it as-is. Reassess the serving basis, ask one clarification question, or use a more credible approximate estimate. "
                "Do not dump multiple raw candidates to the user. You must either ask one concise clarification question or choose one final estimate with a source label. "
                "Do not call meal logging tools. "
                "Return either a concise clarification question or a final nutrition estimate in the user's language. "
                "If the upstream request involves meal logging, still return only the estimate or the clarification question; the main agent will handle prepare_meal_log."
            ),
            "tools": self._nutrition_tools(),
            "skills": [NUTRITION_SKILLS_ROUTE],
        }

        return create_deep_agent(
            model=self.model,
            tools=self.tools,
            skills=[MAIN_SKILLS_ROUTE],
            subagents=[nutrition_specialist],
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
