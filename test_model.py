import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 从 .env 中读取 OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("请在 .env 文件中配置 OPENAI_API_KEY")

# 创建 OpenAI Chat 模型
model = ChatOpenAI(
    model="gpt-4.1-mini",  # 可换成 gpt-4.1 / gpt-4o-mini 等
    api_key=api_key,
)

# 测试一次调用
resp = model.invoke("你好，你是谁？")
print(resp)
