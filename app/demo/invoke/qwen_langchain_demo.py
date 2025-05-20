import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

# 初始化 LangChain 的 Qwen 模型封装
llm = ChatTongyi(
    model="qwen-plus",
    api_key= os.getenv("DASHSCOPE_API_KEY"),
)

# 构建对话消息
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="你是谁？")
]

# 调用模型
response = llm.invoke(messages)

# 输出回复
print("模型回复：", response.content)
