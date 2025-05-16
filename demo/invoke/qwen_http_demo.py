import os
import requests
import json
'''
HTTP方式调用AI
'''
# 从环境变量中读取 API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

# 检查是否成功获取
if not api_key:
    raise ValueError("未设置环境变量 DASHSCOPE_API_KEY，请先 export 或设置到 PyCharm 的 Run Config 中。")

# 请求地址
url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 请求头
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 请求体
payload = {
    "model": "qwen-plus",  # 也可以换成 qwen-turbo 更快
    "input": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"}
        ]
    },
    "parameters": {
        "result_format": "message"
    }
}

# 发起请求
response = requests.post(url, headers=headers, data=json.dumps(payload))

# 打印结果
if response.status_code == 200:
    result = response.json()
    print("模型回复：", result['output']['choices'][0]['message']['content'])
else:
    print(f"请求失败：{response.status_code}")
    print(response.text)
