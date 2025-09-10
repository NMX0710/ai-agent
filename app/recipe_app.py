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
from app.tools.tool_registry import BASE_TOOLS
from app.tools.tool_registry import load_all_tools_with_mcp
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
        # 验证并初始化模型
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请设置后重试")

        self.rag_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
             "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
             "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
             "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
             "\n\n以下是相关的菜谱信息，请基于这些信息回答用户问题：\n{context}"),
            MessagesPlaceholder(variable_name="messages"),

        ])

        # 初始化 LLM
        self.model = ChatTongyi(model="qwen-plus", api_key=api_key)

        # 初始化解析器
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

        # 初始化记忆存储
        self.memory_saver = MemorySaver()

        tools, self._mcp_handle = load_all_tools_with_mcp()
        # 初始化 Agent
        self.agent_executor = create_react_agent(
            self.model,
            tools,
            checkpointer=self.memory_saver,
        )

        # 初始化 embeddings
        embedding_model = DashScopeEmbeddings(
            model="text-embedding-v1",  # 阿里目前推荐的基础 embedding 模型
            dashscope_api_key=api_key
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
        rag_context = "\n\n".join(d.page_content for d in context) if context else "暂无相关菜谱信息"

        # # 使用 RAG template
        # prompt = self.rag_template.invoke({
        #     "messages": messages,
        #     "context": docs_content
        # })

        # 关键：把 RAG 上下文变成系统消息，合并到 messages 里
        system_with_ctx = SystemMessage(
            content=(
                    "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
                    "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
                    "如果用户是减脂人群，推荐低热量高蛋白；若提到特定食材，请结合特点给做法。"
                    "\n\n以下是检索到的菜谱信息（可能为空）：\n" + rag_context
            )
        )
        messages_with_ctx: list[BaseMessage] = [system_with_ctx] + messages

        # response = self.model.invoke(prompt)
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