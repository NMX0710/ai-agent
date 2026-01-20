import logging
import os
from typing import Dict, List, Any

# from langchain_community.chat_models.tongyi import ChatTongyi
# from langchain_community.embeddings import DashScopeEmbeddings

import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel

from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline, RecipeAppState
from app.tools.tool_registry import BASE_TOOLS
from app.tools.tool_registry import load_all_tools_with_mcp
from langgraph.prebuilt import create_react_agent

load_dotenv()

# System prompt
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

# Structured report schema
class RecipeReport(BaseModel):
    title: str
    suggestions: List[str]


class RecipeApp:
    def __init__(self):
        # Validate and initialize the model

        # api_key = os.getenv("DASHSCOPE_API_KEY")
        # if not api_key:
        #     raise RuntimeError("Missing env var DASHSCOPE_API_KEY. Please set it and try again.")
        #
        # self.model = ChatTongyi(model="qwen-plus", api_key=api_key)
        #
        # self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

        # ========= 1) OpenAI Chat model =========
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing environment variable OPENAI_API_KEY. Please configure it in your .env file.")

        # You may switch to gpt-4.1 / gpt-4o-mini, etc.
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=api_key,
            temperature=0.5,
        )

        # ========= 2) Output parser =========
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

        # Memory store (LangGraph checkpointing)
        self.memory_saver = MemorySaver()

        # Optional RAG prompt template
        self.rag_template = ChatPromptTemplate.from_messages([
            ("system",
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

        # Tools & MCP (Agent)
        tools, self._mcp_handle = load_all_tools_with_mcp()
        self.agent_executor = create_react_agent(
            self.model,
            tools,
            checkpointer=self.memory_saver,
        )

        # Embeddings (RAG)
        embedding_model = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=api_key,
        )

        # Prompt template (normal chat)
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Report template
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

        # In-memory chat histories (per chat_id)
        self._chat_histories: Dict[str, List[BaseMessage]] = {}

        # RAG pipeline
        self.rag_pipeline = RecipeAppRAGPipeline(embedding_model, self.model)

        # Build the conversation graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the conversation graph:
          - Use RecipeAppState as the state schema
          - Add 'retrieve' and 'model' nodes
          - Enable MemorySaver for checkpoint-based persistence
        """
        workflow = StateGraph(state_schema=RecipeAppState)
        workflow.add_node("retrieve", self.rag_pipeline.retrieve)
        workflow.add_node("model", self._call_model)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "model")
        workflow.add_edge("model", END)

        return workflow.compile(checkpointer=self.memory_saver)

    def _call_model(self, state: RecipeAppState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Graph node: call the model (Agent + RAG)
          - Use retrieved RAG context
          - Inject context via a SystemMessage and delegate to the ReAct agent
        """
        messages = state.get("messages", [])
        context = state.get("context", [])
        rag_context = "\n\n".join(d.page_content for d in context) if context else "No relevant recipe information found."

        # Inject RAG context as a system message
        system_with_ctx = SystemMessage(
            content=(
                "You are a professional chef knowledgeable in both Chinese and Western cuisine, "
                "with strong cooking experience and nutrition expertise. "
                "Your job is to recommend suitable recipes or practical cooking advice based on the user's "
                "diet preferences, taste, available ingredients, and health needs. "
                "If the user is trying to lose fat, recommend low-calorie, high-protein dishes. "
                "If specific ingredients are mentioned, propose methods based on ingredient characteristics. "
                "\n\nRetrieved recipe information (may be empty):\n" + rag_context + "\n\n"
                "Reply in Chinese unless the user explicitly requests English."
            )
        )
        messages_with_ctx: List[BaseMessage] = [system_with_ctx] + list(messages)

        # Delegate to the agent (preserving checkpointer context)
        response = self.agent_executor.invoke(
            {"messages": messages_with_ctx},
            config=config
        )
        returned_messages: List[BaseMessage] = response["messages"]

        # Use the last AIMessage as the final answer
        last_ai = next((m for m in reversed(returned_messages) if isinstance(m, AIMessage)), None)
        answer = last_ai.content if last_ai else ""

        return {"messages": returned_messages, "answer": answer}

    async def chat(self, chat_id: str, message: str) -> str:
        """
        Chat interface (async):
          - Use the full RecipeAppState
          - Include RAG retrieval
        """
        logging.info("[Chat] chat_id=%s, message=%s", chat_id, message)

        # Load previous history and append the latest user message
        history = self._chat_histories.get(chat_id, [])
        messages = [*history, HumanMessage(content=message)]

        # Initialize state
        state: Dict[str, Any] = {
            "question": message,
            "messages": messages,
            "context": [],
            "answer": None
        }

        config = {"configurable": {"thread_id": chat_id}}

        try:
            out = await self.graph.ainvoke(state, config)
            updated_messages: List[BaseMessage] = out.get("messages", messages)

            # Persist updated chat history
            self._chat_histories[chat_id] = updated_messages

            answer = out.get("answer") or (updated_messages[-1].content if updated_messages else "")
            logging.info("[Chat] response=%s", answer)
            return answer
        except Exception:
            logging.exception("[Chat] Invocation failed.")
            raise

    def generate_report(self, chat_id: str, message: str) -> RecipeReport:
        """
        Generate a structured report (sync):
          - Inject format instructions and query via report_template
          - Call the LLM synchronously
          - Parse JSON output via the Pydantic parser
        """
        logging.info("[Report] chat_id=%s, message=%s", chat_id, message)

        prompt = self.report_template.invoke({"query": message})
        raw = self.model.invoke(prompt)
        report = self.parser.invoke(raw)

        logging.info("[Report] report=%s", report)
        return report
