import os

import dashscope

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Who are you?'},
]

response = dashscope.Generation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen-plus",
    messages=messages,
    result_format='message',
)
print(response)
