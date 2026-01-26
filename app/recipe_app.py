import json
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
import re
from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline, RecipeAppState
from app.tools.tool_registry import BASE_TOOLS
from app.tools.tool_registry import load_all_tools_with_mcp
from langgraph.prebuilt import create_react_agent
from app.models.plan_model import LocalPlanModel


load_dotenv()

PLAN_TAG = "[PLAN]"
EXEC_TAG = "[EXECUTE]"


def strip_tags(text: str, plan_tag: str = "[PLAN]", exec_tag: str = "[EXECUTE]") -> str:
    """
    Remove internal section tags from user-visible output.
    Removes standalone tag lines like:
      [PLAN]
      [EXECUTE]
    and trims surrounding blank lines.
    """
    if not text:
        return text

    # escape in case tags contain regex chars
    plan = re.escape(plan_tag)
    exe = re.escape(exec_tag)

    # remove lines that are exactly the tag (allow surrounding whitespace)
    pattern = rf"(?m)^\s*(?:{plan}|{exe})\s*$\n?"
    cleaned = re.sub(pattern, "", text)

    # remove extra leading blank lines introduced by stripping
    cleaned = cleaned.lstrip("\n").rstrip()
    return cleaned


def render_plan_only(plan: dict) -> str:
    """
    Render a human-readable PLAN-only response (NO EXECUTE).
    """
    constraints = plan.get("constraints", []) or []
    weekly_structure = plan.get("weekly_structure", []) or []
    prep_strategy = plan.get("prep_strategy", []) or []
    questions = plan.get("next_questions", []) or []

    lines = [PLAN_TAG, ""]
    if constraints:
        lines.append("Constraints Summary:")
        lines += [f"- {c}" for c in constraints]
        lines.append("")
    if weekly_structure:
        lines.append("High-Level Weekly Structure:")
        lines += [f"- {s}" for s in weekly_structure]
        lines.append("")
    if prep_strategy:
        lines.append("Prep / Time-Saving Strategy:")
        lines += [f"- {s}" for s in prep_strategy]
        lines.append("")
    if questions:
        lines.append("Clarification Questions:")
        for i, q in enumerate(questions[:3], 1):
            lines.append(f"{i}. {q}")

    return "\n".join(lines).strip()


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
)

# Structured report schema
class RecipeReport(BaseModel):
    title: str
    suggestions: List[str]

class MealPlanPlan(BaseModel):
    need_plan: bool
    # Whether this request requires a plan-first workflow.
    # true  -> produce a structured plan before execution
    # false -> answer directly without a multi-day plan

    constraints: List[str] = []
    # User constraints: time, budget, dietary restrictions, goals, preferences

    weekly_structure: List[str] = []
    # High-level weekly or meal structure (ONLY when need_plan = true)
    # Do NOT include concrete recipes here

    prep_strategy: List[str] = []
    # Meal prep / batching / time-saving strategies

    next_questions: List[str] = []
    # Missing critical information to clarify (max 3)


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

        # ========= Parsers =========
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

        self.plan_parser = PydanticOutputParser(pydantic_object=MealPlanPlan)

        # ======== Plan Node Local Model (Qwen DPO+LoRA) ========
        plan_model_dir = os.getenv("PLAN_MODEL_DIR", "")  # e.g. ./models/plan_qwen_dpo_lora
        plan_base_model = os.getenv("PLAN_BASE_MODEL", "")  # e.g. Qwen/Qwen2.5-7B-Instruct

        self.plan_local_model = None
        if plan_model_dir:
            self.plan_local_model = LocalPlanModel(
                model_dir=plan_model_dir,
                base_model=plan_base_model or None,
                max_new_tokens=int(os.getenv("PLAN_MAX_NEW_TOKENS", "512")),
                temperature=float(os.getenv("PLAN_TEMPERATURE", "0.0")),
            )

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
        Graph structure:
          START -> plan
              -> (plan_only_turn) END
              -> retrieve -> model -> END
        """
        workflow = StateGraph(state_schema=RecipeAppState)

        workflow.add_node("plan", self._plan_node)
        workflow.add_node("retrieve", self.rag_pipeline.retrieve)
        workflow.add_node("model", self._call_model)

        workflow.add_edge(START, "plan")

        def route_after_plan(state: RecipeAppState) -> str:
            return "end" if state.get("plan_only_turn") else "retrieve"

        workflow.add_conditional_edges(
            "plan",
            route_after_plan,
            {
                "end": END,
                "retrieve": "retrieve",
            },
        )

        workflow.add_edge("retrieve", "model")
        workflow.add_edge("model", END)

        return workflow.compile(checkpointer=self.memory_saver)

    def _call_model(self, state: RecipeAppState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Graph node: call the model (Agent + RAG)
          - Use retrieved RAG context
          - Inject context via a SystemMessage and delegate to the ReAct agent
        """

        plan = state.get("plan", None)
        plan_json = json.dumps(plan, ensure_ascii=False) if plan else "null"
        messages = state.get("messages", [])
        context = state.get("context", [])

        def _truncate(text: str, max_chars: int = 1200) -> str:
            return text[:max_chars] + ("..." if len(text) > max_chars else "")

        rag_context = "\n\n".join(
            _truncate(d.page_content) for d in context) if context else "No relevant recipe information found."

        # Inject RAG context as a system message
        system_with_ctx = SystemMessage(
            content=(
                "You are a professional chef with strong nutrition and cooking expertise.\n"
                "Your job is to answer the user by FOLLOWING the approved plan below.\n\n"
                "=== APPROVED PLAN (JSON) ===\n"
                f"{plan_json}\n\n"
                "Execution rules:\n"
                "- If need_plan = true:\n"
                f"  1) First output a brief summary under {PLAN_TAG} (human-readable, concise).\n"
                f"  2) Then output the actionable result under {EXEC_TAG} "
                "(recipes, 7-day meals, shopping list, etc.).\n"
                "- If need_plan = false:\n"
                "  - Do NOT output a weekly plan.\n"
                f"  - Answer directly under {EXEC_TAG}.\n"
                "  - You may ask ONE clarification question if absolutely necessary.\n\n"
                "You may use tools ONLY during EXECUTE if helpful.\n"
                "Retrieved recipe context:\n"
                f"{rag_context}\n"
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
        raw_answer = last_ai.content if last_ai else ""

        # ✅ strip tags for user + future turns (keep internal protocol but don't show it)
        answer = strip_tags(raw_answer, plan_tag=PLAN_TAG, exec_tag=EXEC_TAG)

        # ✅ write back to the actual stored message so history stays clean
        if last_ai is not None:
            last_ai.content = answer

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
            "answer": None,
            "plan_only_turn": False,
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

    def _plan_node(
            self,
            state: RecipeAppState,
            config: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        messages = state.get("messages", [])
        question = state.get("question", "")

        # 保留你原本的 schema instructions（确保输出结构一致）
        format_instructions = self.plan_parser.get_format_instructions()

        # ---- 1) 如果本地 plan 模型存在，优先用它 ----
        if self.plan_local_model is not None:
            history_text = self._format_messages_for_plan(messages)

            local_prompt = (
                "SYSTEM:\n"
                "You are a diet-planning agent.\n"
                "Your task is to decide whether the user's request requires a plan-first approach.\n\n"
                "Rules:\n"
                "1) If the user asks for multi-day, goal-oriented, or multi-constraint planning "
                "(e.g. '7-day meal plan', 'fat loss plan', 'weekly diet'), set need_plan = true.\n"
                "2) If the user asks a single factual or simple question "
                "(e.g. calories of a food, simple substitution, how to cook one dish), "
                "set need_plan = false.\n"
                "3) Do NOT call any tools.\n"
                "4) Do NOT output concrete recipes.\n"
                "5) Ask at most 3 clarification questions.\n"
                "6) Output MUST strictly follow the given JSON schema.\n\n"
                f"{format_instructions}\n\n"
                f"CHAT HISTORY:\n{history_text}\n\n"
                f"USER REQUEST:\n{question}\n\n"
                "OUTPUT JSON ONLY:\n"
            )

            raw_text = self.plan_local_model.generate(local_prompt)

            # 解析：先直接 parse；失败则抽取 JSON 再 parse
            try:
                plan_obj = self.plan_parser.parse(raw_text)
            except Exception:
                m = re.search(r"\{.*\}", raw_text, flags=re.S)
                if not m:
                    raise
                plan_obj = self.plan_parser.parse(m.group(0))

        # ---- 2) 否则 fallback 到 OpenAI（你原本逻辑）----
        else:
            plan_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a diet-planning agent.\n"
                 "Your task is to decide whether the user's request requires a plan-first approach.\n\n"
                 "Rules:\n"
                 "1) If the user asks for multi-day, goal-oriented, or multi-constraint planning "
                 "(e.g. '7-day meal plan', 'fat loss plan', 'weekly diet'), set need_plan = true "
                 "and produce a structured plan.\n"
                 "2) If the user asks a single factual or simple question "
                 "(e.g. calories of a food, simple substitution, how to cook one dish), "
                 "set need_plan = false and DO NOT produce a weekly plan.\n"
                 "3) Do NOT call any tools.\n"
                 "4) Do NOT output concrete recipes.\n"
                 "5) Ask at most 3 clarification questions.\n"
                 "6) Output MUST strictly follow the given JSON schema.\n\n"
                 "{format_instructions}\n"
                 ),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "User request: {question}")
            ])

            prompt = plan_prompt.invoke({
                "messages": messages,
                "question": question,
                "format_instructions": format_instructions
            })

            raw = self.model.invoke(prompt)
            plan_obj = self.plan_parser.invoke(raw)

        plan_dict = plan_obj.model_dump()
        need_plan = bool(plan_dict.get("need_plan"))
        next_qs = plan_dict.get("next_questions", []) or []

        # ✅ PLAN-ONLY TURN: need_plan 且有澄清问题 => 直接结束
        if need_plan and len(next_qs) > 0:
            plan_text = render_plan_only(plan_dict)
            updated_messages = list(messages) + [AIMessage(content=plan_text)]
            return {
                "plan": plan_dict,
                "messages": updated_messages,
                "answer": plan_text,
                "plan_only_turn": True,
            }

        return {
            "plan": plan_dict,
            "plan_only_turn": False,
        }

    def _format_messages_for_plan(self, messages: List[BaseMessage], max_turns: int = 6) -> str:
        recent = messages[-max_turns:] if len(messages) > max_turns else messages
        lines = []
        for m in recent:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines).strip()
