import logging
import os
from typing import Dict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

#TODO: Add LangSmith to keep track of agents later

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
    "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
    "在开场时，向用户表明你的厨师身份，并邀请他们描述自己的需求。"
    "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
    "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
    "引导用户尽量具体描述需求，你再基于这些信息提供实用、详细、有趣的菜谱推荐。"
)

class RecipeApp():
    def __init__(self):
        # Initialize the model
        self.graph = self._build_graph()
        self.model = ChatTongyi(
            model="qwen-plus",
            api_key= os.getenv("DASHSCOPE_API_KEY"),
        )
        self.prompt_template = prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def _call_model(self, state: MessagesState):
        prompt = self.prompt_template.invoke(state)
        response = self.model.invoke(prompt)
        return {"messages": response}

    def _build_graph(self):
        # Define a new graph
        workflow = StateGraph(state_schema=MessagesState)
        # Define the (single) node in the graph
        workflow.add_edge(START, "model")
        workflow.add_node("model", self._call_model)
        # Add memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def chat(self, chat_id: str, message: str) -> str:
        input_messages = [HumanMessage(content=message)]
        config = {"configurable": {"thread_id": chat_id}}
        output = await self.graph.ainvoke({"messages": input_messages}, config)
        return output["messages"][-1].content








