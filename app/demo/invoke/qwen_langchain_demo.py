import os

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatTongyi(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who are you?"),
]

response = llm.invoke(messages)
print("Model response:", response.content)
