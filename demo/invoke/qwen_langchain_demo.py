import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

# 确保你已经设置了环境变量 DASHSCOPE_API_KEY
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未设置环境变量 DASHSCOPE_API_KEY")

# 初始化 LangChain 的 Qwen 模型封装
llm = ChatTongyi(
    model="qwen-plus",  # 可以改成 "qwen-turbo"
    dashscope_api_key=api_key,
    result_format="message"  # 返回格式与 openai 一致
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
