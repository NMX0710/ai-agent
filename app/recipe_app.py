import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline, RecipeAppState
from app.tools.tool_registry import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are an expert chef with strong knowledge of Chinese and Western cooking, nutrition, "
    "meal planning, and practical home cooking. Help the user with recipe ideas, cooking suggestions, "
    "and meal recommendations based on their dietary goals, ingredients, flavor preferences, time constraints, "
    "and health needs. If the user wants fat-loss meals, prioritize high-protein, lower-calorie options. "
    "If the user mentions a specific ingredient, explain realistic and practical ways to use it. "
    "Always answer in English."
)


class RecipeReport(BaseModel):
    title: str
    suggestions: List[str]


class RecipeApp:
    def __init__(self):
        self.rag_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert chef with strong knowledge of cooking and nutrition. "
                "Use the retrieved recipe context when it is relevant, but keep the answer grounded, practical, and clear. "
                "Always answer in English.\n\nRetrieved recipe context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.model, embedding_model, self.model_provider = self._build_model_and_embeddings()
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)
        self.memory_saver = MemorySaver()

        self.agent_executor = create_react_agent(
            self.model,
            ALL_TOOLS,
            checkpointer=self.memory_saver,
        )

        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.report_template = PromptTemplate(
            template=(
                "Generate a short report after each conversation as a list of suggestions.\n"
                "{format_instructions}\n{query}\n"
            ),
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
            },
        )

        self.rag_pipeline = RecipeAppRAGPipeline(embedding_model, self.model)
        self.graph = self._build_graph()

    def _build_model_and_embeddings(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Set it before running the app.")

        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=openai_api_key,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        )
        embedding_model = (
            os.getenv("OPENAI_EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBED_MODEL")
            or "text-embedding-3-small"
        )
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key,
        )
        return model, embeddings, "openai"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(state_schema=RecipeAppState)
        workflow.add_node("retrieve", self.rag_pipeline.retrieve)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "model")
        workflow.add_edge("model", END)
        return workflow.compile(checkpointer=self.memory_saver)

    def _call_model(
        self,
        state: RecipeAppState,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        messages = state.get("messages", [])
        context = state.get("context", [])
        messages_with_ctx, _ = self._build_agent_messages(messages, context)

        response = self.agent_executor.invoke(
            {"messages": messages_with_ctx},
            config=config,
        )
        returned_messages: List[BaseMessage] = response["messages"]
        for message in returned_messages:
            try:
                message.pretty_print()
            except Exception:
                pass

        last_ai = next((m for m in reversed(returned_messages) if isinstance(m, AIMessage)), None)
        answer = last_ai.content if last_ai else ""
        return {"messages": returned_messages, "answer": answer}

    def _build_agent_messages(
        self,
        messages: List[BaseMessage],
        context: List[Document],
    ) -> tuple[List[BaseMessage], str]:
        rag_context = (
            "\n\n".join(document.page_content for document in context)
            if context else "No relevant recipe context retrieved."
        )

        system_with_ctx = SystemMessage(
            content=(
                "You are an expert chef with strong knowledge of nutrition, cooking methods, and home meal planning. "
                "Always answer in English. Use the retrieved recipe context when it helps, but do not invent facts that are not supported by the context or the user's request. "
                "If the user asks for fat-loss meals, prioritize lower-calorie, high-protein options. "
                "If the user mentions a specific ingredient, explain practical ways to cook it."
                "\n\nRetrieved recipe context (may be empty):\n" + rag_context
            )
        )
        return [system_with_ctx, *messages], rag_context

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [RecipeApp._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): RecipeApp._jsonable(item) for key, item in value.items()}
        return str(value)

    @classmethod
    def _message_to_dict(cls, message: BaseMessage) -> Dict[str, Any]:
        role = message.type
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"

        payload = {
            "role": role,
            "content": cls._jsonable(message.content),
        }

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = cls._jsonable(tool_calls)

        name = getattr(message, "name", None)
        if name:
            payload["name"] = name

        return payload

    @staticmethod
    def _document_to_dict(document: Document) -> Dict[str, Any]:
        return {
            "page_content": document.page_content,
            "metadata": RecipeApp._jsonable(document.metadata),
        }

    @staticmethod
    def _coerce_history_messages(history: Optional[List[Dict[str, Any]]]) -> List[BaseMessage]:
        if not history:
            return []

        messages: List[BaseMessage] = []
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                raise ValueError(f"Unsupported history role: {role}")
        return messages

    def generate_dpo_record(
        self,
        sample_id: str,
        user_input: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        history_messages = self._coerce_history_messages(conversation_history)
        messages = [*history_messages, HumanMessage(content=user_input)]

        state: RecipeAppState = {
            "question": user_input,
            "messages": messages,
            "context": [],
            "answer": None,
        }
        retrieved_state = self.rag_pipeline.retrieve(state, top_k=4)
        context = retrieved_state.get("context", [])
        messages_with_ctx, rag_context = self._build_agent_messages(messages, context)

        response = self.agent_executor.invoke(
            {"messages": messages_with_ctx},
            config={"configurable": {"thread_id": chat_id or sample_id}},
        )
        returned_messages: List[BaseMessage] = response["messages"]
        last_ai = next((m for m in reversed(returned_messages) if isinstance(m, AIMessage)), None)

        return {
            "sample_id": sample_id,
            "user_input": user_input,
            "retrieved_context": [self._document_to_dict(doc) for doc in context],
            "retrieved_context_text": rag_context,
            "agent_input": {
                "messages": [self._message_to_dict(message) for message in messages_with_ctx],
            },
            "messages_before_agent_call": [
                self._message_to_dict(message) for message in messages_with_ctx
            ],
            "returned_messages": [
                self._message_to_dict(message) for message in returned_messages
            ],
            "chosen": last_ai.content if last_ai else "",
            "metadata": metadata or {},
            "model_provider": self.model_provider,
            "tool_names": [getattr(tool, "name", str(tool)) for tool in ALL_TOOLS],
        }

    async def chat(self, chat_id: str, message: str) -> str:
        logging.info(f"[Chat] chat_id={chat_id}, message={message}")

        state = {
            "question": message,
            "messages": [HumanMessage(content=message)],
            "context": [],
            "answer": None,
        }

        config = {"configurable": {"thread_id": chat_id}}

        try:
            output = await self.graph.ainvoke(state, config)
            answer = output.get("answer") or output["messages"][-1].content
            logging.info(f"[Chat] response={answer}")
            return answer
        except Exception:
            logging.exception("Chat call failed")
            raise

    def generate_report(self, chat_id: str, message: str) -> RecipeReport:
        logging.info(f"[Report] chat_id={chat_id}, message={message}")
        prompt = self.report_template.invoke({"query": message})
        raw = self.model.invoke(prompt)
        report = self.parser.invoke(raw)
        logging.info(f"[Report] report: {report}")
        return report
