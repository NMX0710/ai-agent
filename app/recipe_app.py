import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.memory import PostgresMemoryStore, build_memory_candidate
from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline, RecipeAppState
from app.settings import (
    DATABASE_URL,
    LONG_TERM_MEMORY_ENABLED,
    MEMORY_MAX_RECORDS_PER_USER,
    MEMORY_RETRIEVE_TOP_K,
    MEMORY_RETENTION_DAYS,
)
from app.tools.tool_registry import load_all_tools_with_mcp

load_dotenv()

SYSTEM_PROMPT = (
    "You are a professional chef knowledgeable in both Chinese and Western cuisine, "
    "with strong cooking experience and nutrition expertise. "
    "Your job is to recommend suitable recipes or practical cooking advice based on the user's "
    "diet preferences, taste, available ingredients, and health needs. "
    "At the start, introduce yourself as a chef and invite the user to describe their needs. "
    "If the user is trying to lose fat, recommend low-calorie, high-protein, light dishes. "
    "If the user wants to improve everyday family meals, suggest easy-to-cook, nutritious home-style dishes. "
    "If the user mentions specific ingredients (e.g., chicken, tomatoes), recommend appropriate techniques "
    "based on the ingredient characteristics. "
    "Encourage the user to be as specific as possible so you can provide practical, detailed, and fun recommendations. "
    "Reply in Chinese unless the user explicitly requests English."
)


class RecipeReport(BaseModel):
    title: str
    suggestions: List[str]


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
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)
        self.memory_saver = MemorySaver()

        self.rag_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional chef knowledgeable in both Chinese and Western cuisine, "
                "with strong cooking experience and nutrition expertise. "
                "Your job is to recommend suitable recipes or practical cooking advice based on the user's "
                "diet preferences, taste, available ingredients, and health needs. "
                "If the user is trying to lose fat, recommend low-calorie, high-protein, light dishes. "
                "If the user wants to improve everyday family meals, suggest easy-to-cook, nutritious home-style dishes. "
                "If the user mentions specific ingredients (e.g., chicken, tomatoes), recommend appropriate techniques "
                "based on the ingredient characteristics. "
                "\n\nBelow is relevant recipe information. Use it to answer the user:\n{context}\n\n"
                "Reply in Chinese unless the user explicitly requests English."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        tools, self._mcp_handle = load_all_tools_with_mcp()
        self.agent_executor = create_react_agent(
            self.model,
            tools,
            checkpointer=self.memory_saver,
        )

        embedding_model = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=api_key,
        )

        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.report_template = PromptTemplate(
            template=(
                "After each conversation, generate a short report consisting of a list of suggestions.\n"
                "{format_instructions}\n"
                "{query}\n"
            ),
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        self._chat_histories: Dict[str, List[BaseMessage]] = {}

        self.memory_store = PostgresMemoryStore(
            dsn=DATABASE_URL,
            retention_days=MEMORY_RETENTION_DAYS,
            max_records_per_user=MEMORY_MAX_RECORDS_PER_USER,
        )
        if LONG_TERM_MEMORY_ENABLED:
            self.memory_store.initialize()
        else:
            logging.info("[MemoryStore] LONG_TERM_MEMORY_ENABLED=0 -> disabled")

        self.rag_pipeline = RecipeAppRAGPipeline(embedding_model, self.model)
        self.graph = self._build_graph()

    @staticmethod
    def _history_key(user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(state_schema=RecipeAppState)
        workflow.add_node("retrieve", self.rag_pipeline.retrieve)
        workflow.add_node("model", self._call_model)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "model")
        workflow.add_edge("model", END)

        return workflow.compile(checkpointer=self.memory_saver)

    def _call_model(self, state: RecipeAppState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        messages = state.get("messages", [])
        context = state.get("context", [])
        long_term_memories = state.get("memory", [])

        rag_context = "\n\n".join(d.page_content for d in context) if context else "No relevant recipe information found."
        memory_context = "\n".join(long_term_memories) if long_term_memories else "No long-term user memory available."

        system_with_ctx = SystemMessage(
            content=(
                "You are a professional chef knowledgeable in both Chinese and Western cuisine, "
                "with strong cooking experience and nutrition expertise. "
                "Your job is to recommend suitable recipes or practical cooking advice based on the user's "
                "diet preferences, taste, available ingredients, and health needs. "
                "If the user is trying to lose fat, recommend low-calorie, high-protein dishes. "
                "If specific ingredients are mentioned, propose methods based on ingredient characteristics. "
                "\n\nRetrieved recipe information (may be empty):\n"
                + rag_context
                + "\n\nRetrieved user memory snippets:\n"
                + memory_context
                + "\n\nReply in Chinese unless the user explicitly requests English."
            )
        )
        messages_with_ctx: List[BaseMessage] = [system_with_ctx] + list(messages)

        response = self.agent_executor.invoke(
            {"messages": messages_with_ctx},
            config=config
        )
        returned_messages: List[BaseMessage] = response["messages"]

        last_ai = next((m for m in reversed(returned_messages) if isinstance(m, AIMessage)), None)
        answer = last_ai.content if last_ai else ""

        return {"messages": returned_messages, "answer": answer}

    def _save_long_term_memory(self, user_id: str, chat_id: str, user_message: str, answer: str) -> None:
        candidate = build_memory_candidate(user_message=user_message, assistant_message=answer)
        if not candidate:
            return

        self.memory_store.upsert_memory(
            user_id=user_id,
            memory_key=candidate.memory_key,
            summary=candidate.summary,
            keywords=candidate.keywords,
            source_chat_id=chat_id,
        )

    async def chat(self, chat_id: str, message: str, user_id: str = "anonymous-user") -> str:
        logging.info("[Chat] user_id=%s, chat_id=%s, message=%s", user_id, chat_id, message)

        history_key = self._history_key(user_id, chat_id)
        history = self._chat_histories.get(history_key, [])
        messages = [*history, HumanMessage(content=message)]

        memory_items = self.memory_store.retrieve(
            user_id=user_id,
            query=message,
            limit=MEMORY_RETRIEVE_TOP_K,
        )
        memory_snippets = [
            f"- {item.summary} (keywords: {', '.join(item.keywords[:6])})"
            for item in memory_items
        ]

        state: Dict[str, Any] = {
            "question": message,
            "messages": messages,
            "context": [],
            "memory": memory_snippets,
            "answer": None
        }

        config = {"configurable": {"thread_id": history_key}}

        try:
            out = await self.graph.ainvoke(state, config)
            updated_messages: List[BaseMessage] = out.get("messages", messages)
            self._chat_histories[history_key] = updated_messages

            answer = out.get("answer") or (updated_messages[-1].content if updated_messages else "")
            self._save_long_term_memory(user_id, chat_id, message, answer)

            logging.info("[Chat] response=%s", answer)
            return answer
        except Exception:
            logging.exception("[Chat] Invocation failed.")
            raise

    def generate_report(self, chat_id: str, message: str) -> RecipeReport:
        logging.info("[Report] chat_id=%s, message=%s", chat_id, message)

        prompt = self.report_template.invoke({"query": message})
        raw = self.model.invoke(prompt)
        report = self.parser.invoke(raw)

        logging.info("[Report] report=%s", report)
        return report
