import logging
import os
from typing import Dict, List
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, BasePromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from app.wrappers.logger import log_wrapper

#TODO: Fix the error and finish structured output


SYSTEM_PROMPT = (
    "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
    "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
    "在开场时，向用户表明你的厨师身份，并邀请他们描述自己的需求。"
    "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
    "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
    "引导用户尽量具体描述需求，你再基于这些信息提供实用、详细、有趣的菜谱推荐。"
)


class RecipeReport(BaseModel):
    title: str
    suggestions: List[str]

class RecipeApp:
    def __init__(self):
        # Initialize the model
        self.prompt_template = None
        self.graph = self._build_graph()
        self.model = ChatTongyi(
            model="qwen-plus",
            api_key= os.getenv("DASHSCOPE_API_KEY"),
        )
        self.parser = PydanticOutputParser(pydantic_object=RecipeReport)

    """
    Graph 节点: 调用模型
    - 输入是从graph获取的state
    - 根据当前 prompt_template 生成 prompt
    - 调用 LLM 并获取响应
    - 更新并返回消息历史
    """
    def _call_model(self, state: MessagesState):
        """
        messages 是一个包含对话历史的列表
        每个元素都是一个 BaseMessage 子类的对象，比如：
        HumanMessage(content="你好，我喜欢吃鸡胸肉")
        AIMessage(content="你好！根据你的喜好...")
        """
        messages = state["messages"]
        # 将系统 + 历史对话拼到一起
        prompt = self.prompt_template.invoke({"messages": messages})
        # 调用模型
        response = self.model.invoke(prompt)
        # 更新消息历史
        new_messages = messages + [response]
        return {"messages": new_messages}

    """
    构建对话graph
    - 使用 MessagesState 作为 state schema
    - 注册 _call_model 节点
    - 添加 MemorySaver 持久化上下文
    """
    def _build_graph(self):
        # TODO:没有规定上下文记忆容量

        # Define a new graph
        workflow = StateGraph(state_schema= MessagesState)

        # Define the nodes in the graph
        workflow.add_node("model", self._call_model)

        # 链接顺序：START → model
        workflow.add_edge(START, "model")

        # Add memory
        # TODO：当前的储存器是内存级别的，所以在重启之后之前的对话会丢失，我们希望实现上下文持久化
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    """
    对话
    """
    async def chat(self, chat_id: str, message: str) -> str:
        # state = ReportState(messages=[HumanMessage(content=message)], report=RecipeReport(title="", suggestions=[]))
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        state = {"messages": [HumanMessage(content=message)]}
        config = {"configurable": {"thread_id": chat_id}}
        out = await self.graph.ainvoke(state, config)
        # 返回原始文本回答
        return out["messages"][-1].content

    """
    生成结构化报告:
    - 使用 PromptTemplate 注入报告指令
    - 调用模型并用 parser 解析 JSON
    - 绕过graph调用模型（缺失上下文信息）
    """
    def generate_report(self, chat_id: str, message: str) -> RecipeReport:

        parser = PydanticOutputParser(pydantic_object=RecipeReport)

        self.prompt_template = PromptTemplate(
            template="每次对话后都要生成报告，内容为建议列表\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        prompt_and_model = self.prompt_template | self.model
        config = {"configurable": {"thread_id": chat_id}}
        out = prompt_and_model.invoke({"query": message}, config)
        out = parser.invoke(out)
        return out















