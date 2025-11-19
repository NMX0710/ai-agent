import boto3
import json
import uuid

# ---- 1. 创建 Bedrock AgentCore 客户端 ----
client = boto3.client("bedrock-agentcore", region_name="us-east-1")

# ---- 2. 你的 Runtime ARN ----
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:658394313796:runtime/recipe_agent_runtime-ohVyInCiEq"

# ---- 3. 构造 payload：要符合你 FastAPI /invocations 的结构 ----
payload = {
    "input": {
        "prompt": "帮我规划一份高蛋白低脂的晚餐"
    }
}

# ---- 4. 调用 Runtime ----
response = client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    runtimeSessionId=str(uuid.uuid4()),
    payload=json.dumps(payload).encode("utf-8"),
    qualifier="DEFAULT",
)

print("Raw response dict keys:", response.keys())
print("contentType:", response.get("contentType"))

raw_body = ""

# 如果是 JSON 返回（最常见的情况）
if response.get("contentType") == "application/json":
    chunks = []
    for chunk in response.get("response", []):
        # 每个 chunk 是 bytes
        chunks.append(chunk.decode("utf-8"))
    raw_body = "".join(chunks)

# 如果是流式 text/event-stream（可选处理方式）
elif "text/event-stream" in (response.get("contentType") or ""):
    chunks = []
    # 这里的 response["response"] 通常是一个支持 iter_lines 的对象
    for line in response["response"].iter_lines(chunk_size=10):
        if line:
            line = line.decode("utf-8")
            print("STREAM LINE:", line)
            if line.startswith("data: "):
                line = line[6:]
            chunks.append(line)
    raw_body = "\n".join(chunks)

# 其它类型就直接打印整个 response 方便调试
else:
    print("Unknown contentType, full response:")
    print(response)

print("\nRaw body:", raw_body)

# 尝试解析 JSON
try:
    data = json.loads(raw_body)
    # 这里根据你容器返回的字段名来取，比如 "response" 或 "output" 等
    print("\nParsed JSON:", data)
except json.JSONDecodeError:
    print("\n(返回的不是合法 JSON，需要去看容器日志排查)")
