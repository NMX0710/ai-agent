import logging
import os
from typing import Dict, List, Optional, TypedDict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel
from app.rag.recipe_app_rag_pipeline import RecipeAppRAGPipeline

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

class RecipeAppState(TypedDict):
    question: str
    context: List[Document]           # RAG 检索到的内容
    messages: List[BaseMessage]       # 历史消息记录（LangGraph 自带）
    answer: Optional[str]             # 最终答案


#如果只是一个小型的project，init参数可以不需要，因为我们只需要一个RecipeApp实例
class RecipeApp:
    def __init__(self):
        # 验证并初始化模型
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY，请设置后重试")
        model = ChatTongyi(model="qwen-plus", api_key=api_key)
        self.model = model

        # 初始化解析器
        parser = PydanticOutputParser(pydantic_object=RecipeReport)
        self.parser = parser

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
        # 构建对话图
        self.graph = self._build_graph()

        # 初始化rag pipeline
        embedding_model = DashScopeEmbeddings(
            model="text-embedding-v1",  # 阿里目前推荐的基础 embedding 模型
            dashscope_api_key=api_key
        )
        self.rag_pipeline = RecipeAppRAGPipeline(embedding_model)

    def _build_graph(self) -> StateGraph:
        """
        构建对话graph
        - 使用 MessagesState 作为 state schema
        - 注册 _call_model 节点
        - 添加 MemorySaver 持久化上下文
        """
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow.compile(checkpointer=self.memory_saver)

    def _call_model(self, state: MessagesState) -> Dict[str, List[BaseMessage]]:
        """
        Graph 节点: 调用模型
        - 输入是从graph获取的state
        - 根据当前 prompt_template 生成 prompt
        - 调用 LLM 并获取响应
        - 更新并返回消息历史
        """
        messages = state["messages"]

        # TODO：RAG 检索补充
        prompt = self.chat_template.invoke({"messages": messages})
        logging.info(f"[Model] prompt: {prompt}")
        response = self.model.invoke(prompt)
        logging.info(f"[Model] response: {response.content}")
        new_messages = messages + [response]
        return {"messages": new_messages}


    async def chat(self, chat_id: str, message: str) -> str:
        """
        对话接口（异步）
        """
        logging.info(f"[Chat] chat_id={chat_id}, message={message}")
        state = {"messages": [HumanMessage(content=message)]}
        config = {"configurable": {"thread_id": chat_id}}
        try:
            out = await self.graph.ainvoke(state, config)
            answer = out["messages"][-1].content
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
        logging.info(f"[Report] prompt: {prompt}")
        raw = self.model.invoke(prompt)
        logging.info(f"[Report] raw: {raw}")
        report = self.parser.invoke(raw)
        logging.info(f"[Report] report: {report}")
        return report
