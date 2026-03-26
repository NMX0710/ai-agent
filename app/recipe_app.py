import json
import logging
import os
from typing import Dict, List, Optional, TypedDict, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline, RecipeAppState
from app.tools.tool_registry import ALL_TOOLS
from langgraph.prebuilt import create_react_agent

# 系统提示
SYSTEM_PROMPT = (
    "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
    "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
    "在开场时，向用户表明你的厨师身份，并邀请他们描述自己的需求。"
    "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
    "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
    "引导用户尽量具体描述需求，你再基于这些信息提供实用、详细、有趣的菜谱推荐。"
)


# 定义结构化报告模型
class RecipeReport(BaseModel):
    title: str
    suggestions: List[str]


class RecipeApp:
    def __init__(self):
        self.rag_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
             "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
             "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
             "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
             "\n\n以下是相关的菜谱信息，请基于这些信息回答用户问题：\n{context}"),
            MessagesPlaceholder(variable_name="messages"),

        ])

        self.model, embedding_model, self.model_provider = self._build_model_and_embeddings()

        # 初始化解析器
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

        # 初始化记忆存储
        self.memory_saver = MemorySaver()

        # 初始化 Agent
        self.agent_executor = create_react_agent(
            self.model, ALL_TOOLS,
            checkpointer=self.memory_saver,
        )

        # 准备 Prompt Templates
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.report_template = PromptTemplate(
            template=(
                "每次对话后都要生成报告，内容为建议列表\n"
                "{format_instructions}\n{query}\n"
            ),
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )


        # 初始化 RAG pipeline，传入 model
        self.rag_pipeline = RecipeAppRAGPipeline(embedding_model, self.model)

        # 构建对话图
        self.graph = self._build_graph()

    def _build_model_and_embeddings(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if openai_api_key:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                api_key=openai_api_key,
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            )
            if not dashscope_api_key:
                raise RuntimeError("使用 OPENAI_API_KEY 时也需要 DASHSCOPE_API_KEY 来构建现有 RAG 索引。")
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v1",
                dashscope_api_key=dashscope_api_key
            )
            return model, embeddings, "openai"

        if not dashscope_api_key:
            raise RuntimeError("缺少环境变量 OPENAI_API_KEY 或 DASHSCOPE_API_KEY，请设置后重试")

        model = ChatTongyi(model="qwen-plus", api_key=dashscope_api_key)
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=dashscope_api_key
        )
        return model, embeddings, "dashscope"

    def _build_graph(self) -> StateGraph:
        """
        构建对话graph
        - 使用 RecipeAppState 作为 state schema
        - 添加 retrieve 和 model 节点
        - 添加 MemorySaver 持久化上下文
        """
        workflow = StateGraph(state_schema=RecipeAppState)

        # 添加节点
        workflow.add_node("retrieve", self.rag_pipeline.retrieve)
        workflow.add_node("model", self._call_model)

        # 添加边
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "model")
        workflow.add_edge("model", END)

        return workflow.compile(checkpointer=self.memory_saver)


    def _call_model(self, state: RecipeAppState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Graph 节点: 调用模型
        - 使用 RAG 检索到的上下文
        - 生成回复并更新状态
        """
        messages = state.get("messages", [])
        context = state.get("context", [])
        messages_with_ctx, _ = self._build_agent_messages(messages, context)

        response = self.agent_executor.invoke(
            {"messages": messages_with_ctx},
            config=config  # 关键：把 thread_id 等透传，以启用 checkpointer
        )
        returned_messages: List[BaseMessage] = response["messages"]
        # （可选）调试：打印每步
        for m in returned_messages:
            try:
                m.pretty_print()
            except Exception:
                pass

        # 取最后一条 AIMessage 作为最终答案
        last_ai = next((m for m in reversed(returned_messages) if isinstance(m, AIMessage)), None)
        answer = last_ai.content if last_ai else ""

        return {"messages": returned_messages, "answer": answer}

    def _build_agent_messages(
        self,
        messages: List[BaseMessage],
        context: List[Document],
    ) -> tuple[List[BaseMessage], str]:
        rag_context = "\n\n".join(d.page_content for d in context) if context else "暂无相关菜谱信息"

        system_with_ctx = SystemMessage(
            content=(
                "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
                "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
                "如果用户是减脂人群，推荐低热量高蛋白；若提到特定食材，请结合特点给做法。"
                "\n\n以下是检索到的菜谱信息（可能为空）：\n" + rag_context
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
        """
        对话接口（异步）
        - 使用完整的 RecipeAppState
        - 包含 RAG 检索
        """
        logging.info(f"[Chat] chat_id={chat_id}, message={message}")

        # 初始化 state，包含所有必要的字段
        state = {
            "question": message,
            "messages": [HumanMessage(content=message)],
            "context": [],
            "answer": None
        }

        config = {"configurable": {"thread_id": chat_id}}

        try:
            out = await self.graph.ainvoke(state, config)
            answer = out.get("answer") or out["messages"][-1].content
            logging.info(f"[Chat] response={answer}")
            return answer
        except Exception:
            logging.exception("chat 调用失败")
            raise

    def generate_report(self, chat_id: str, message: str) -> RecipeReport:
        """
        生成结构化报告（同步）
        - 使用 report_template 注入格式指令和查询
        - 同步调用 LLM
        - 用 parser 解析 JSON 输出
        """
        logging.info(f"[Report] chat_id={chat_id}, message={message}")
        prompt = self.report_template.invoke({"query": message})
        raw = self.model.invoke(prompt)
        report = self.parser.invoke(raw)
        logging.info(f"[Report] report: {report}")
        return report
