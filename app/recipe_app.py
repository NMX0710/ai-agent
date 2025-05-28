import logging
import os
from typing import Dict, List, Optional, TypedDict, Any
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel
from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline, RecipeAppState

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

        # 初始化 LLM
        self.model = ChatTongyi(model="qwen-plus", api_key=api_key)

        # 初始化解析器
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

        # 初始化记忆存储
        self.memory_saver = MemorySaver()

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

        self.rag_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
             "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
             "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
             "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
             "\n\n以下是相关的菜谱信息，请基于这些信息回答用户问题：\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # 初始化 embeddings
        embedding_model = DashScopeEmbeddings(
            model="text-embedding-v1",  # 阿里目前推荐的基础 embedding 模型
            dashscope_api_key=api_key
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
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("model", self._call_model)

        # 添加边
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "model")
        workflow.add_edge("model", END)

        return workflow.compile(checkpointer=self.memory_saver)

    def _retrieve_context(self, state: RecipeAppState) -> Dict[str, Any]:
        """
        Graph 节点: 调用 RAG pipeline 进行检索
        """
        # 如果 state 中没有 question，从最新的消息中提取
        if not state.get("question"):
            messages = state.get("messages", [])
            if messages:
                # 获取最新的用户消息作为问题
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        state["question"] = msg.content
                        break

        # 调用 RAG pipeline 的 retrieve 方法
        result = self.rag_pipeline.retrieve(state, top_k=4)
        return result

    def _call_model(self, state: RecipeAppState) -> Dict[str, Any]:
        """
        Graph 节点: 调用模型
        - 使用 RAG 检索到的上下文
        - 生成回复并更新状态
        """
        messages = state.get("messages", [])
        context = state.get("context", [])

        # 构建上下文内容
        if context:
            docs_content = "\n\n".join(doc.page_content for doc in context)
        else:
            docs_content = "暂无相关菜谱信息"

        # 使用 RAG template
        prompt = self.rag_template.invoke({
            "messages": messages,
            "context": docs_content
        })

        logging.info(f"[Model] prompt: {prompt}")
        response = self.model.invoke(prompt)
        logging.info(f"[Model] response: {response.content}")

        # 更新消息和答案
        new_messages = messages + [response]
        return {
            "messages": new_messages,
            "answer": response.content
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