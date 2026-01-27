import boto3
import json
import uuid

client = boto3.client("bedrock-agentcore", region_name="us-east-1")

RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:658394313796:"
    "runtime/recipe_agent_runtime-ohVyInCiEq"
)

payload = {
    "input": {
        "prompt": "Plan a 7-day fat-loss meal plan."
    }
}

response = client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    runtimeSessionId=str(uuid.uuid4()),
    payload=json.dumps(payload).encode("utf-8"),
    qualifier="DEFAULT",
)

print("Raw response dict keys:", response.keys())
print("contentType:", response.get("contentType"))

raw_body = ""

ct = response.get("contentType") or ""
body_obj = response.get("response")

# ---- robust body reading ----
if body_obj is None:
    print("No 'response' field found. Full response:")
    print(response)
else:
    # Case A: list of bytes chunks
    if isinstance(body_obj, list):
        raw_body = b"".join(body_obj).decode("utf-8", errors="replace")

    # Case B: streaming-like object
    elif hasattr(body_obj, "read"):
        raw_body = body_obj.read().decode("utf-8", errors="replace")

    # Case C: iter_lines streaming (SSE)
    elif hasattr(body_obj, "iter_lines"):
        lines = []
        for line in body_obj.iter_lines():
            if line:
                line = line.decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    line = line[6:]
                lines.append(line)
        raw_body = "\n".join(lines)

    else:
        raw_body = str(body_obj)

print("\nRaw body:")
print(raw_body)

try:
    data = json.loads(raw_body)
    print("\nParsed JSON:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if isinstance(data, dict) and "output" in data:
        print("\nModel output:")
        print(data["output"])
except json.JSONDecodeError:
    print("\n(The response is not valid JSON. Check the runtime/container logs for debugging.)")
