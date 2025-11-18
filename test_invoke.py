import boto3
import json
import uuid

# ---- 1. 创建 Bedrock AgentCore 客户端 ----
client = boto3.client("bedrock-agentcore", region_name="us-east-1")

# ---- 2. 你的 Runtime ARN（从控制台复制完整的那一串）----
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:658394313796:runtime/recipe_agent_runtime-ohVyInCiEq"  # 用你的实际 ARN

# ---- 3. 构造 payload：要符合你 FastAPI /invocations 的结构 ----
payload = {
    "input": {
        "prompt": "帮我规划一份高蛋白低脂的晚餐"
    }
}

# ---- 4. 调用 Runtime ----
response = client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    runtimeSessionId=str(uuid.uuid4()),      # 随便生成一个 32+ 字符的 session id
    payload=json.dumps(payload).encode("utf-8"),
    qualifier="DEFAULT",                     # Runtime 默认版本名
)

# ---- 5. 读取并打印返回结果 ----
raw_body = response["body"].read().decode("utf-8")
print("Raw response:", raw_body)

try:
    data = json.loads(raw_body)
    print("\nParsed output:", data.get("output"))
except json.JSONDecodeError:
    print("\n(返回的不是合法 JSON，需要去看容器日志排查)")
