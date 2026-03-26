import json
import os

import requests

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("DASHSCOPE_API_KEY is not set. Export it before running this demo.")

url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "qwen-plus",
    "input": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
    },
    "parameters": {"result_format": "message"},
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
if response.status_code == 200:
    result = response.json()
    print("Model response:", result['output']['choices'][0]['message']['content'])
else:
    print(f"Request failed: {response.status_code}")
    print(response.text)
