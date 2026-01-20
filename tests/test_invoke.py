import boto3
import json
import uuid

# ------------------------------------------------------------
# 1. Create the Bedrock AgentCore client
# ------------------------------------------------------------
client = boto3.client(
    "bedrock-agentcore",
    region_name="us-east-1",
)

# ------------------------------------------------------------
# 2. Agent Runtime ARN
# ------------------------------------------------------------
RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:658394313796:"
    "runtime/recipe_agent_runtime-ohVyInCiEq"
)

# ------------------------------------------------------------
# 3. Build the request payload
#    This payload must match the structure expected by your
#    FastAPI /invocations endpoint inside the runtime container.
# ------------------------------------------------------------
payload = {
    "input": {
        "prompt": "Help me plan a high-protein, low-fat dinner"
    }
}

# ------------------------------------------------------------
# 4. Invoke the Agent Runtime
# ------------------------------------------------------------
response = client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    runtimeSessionId=str(uuid.uuid4()),  # Unique session ID per request
    payload=json.dumps(payload).encode("utf-8"),
    qualifier="DEFAULT",
)

# ------------------------------------------------------------
# 5. Inspect basic response metadata
# ------------------------------------------------------------
print("Raw response dict keys:", response.keys())
print("contentType:", response.get("contentType"))

raw_body = ""

# ------------------------------------------------------------
# 6. Handle JSON response (most common case)
# ------------------------------------------------------------
if response.get("contentType") == "application/json":
    chunks = []
    for chunk in response.get("response", []):
        # Each chunk is returned as bytes
        chunks.append(chunk.decode("utf-8"))
    raw_body = "".join(chunks)

# ------------------------------------------------------------
# 7. Handle streaming responses (text/event-stream)
# ------------------------------------------------------------
elif "text/event-stream" in (response.get("contentType") or ""):
    chunks = []
    # response["response"] usually supports iter_lines()
    for line in response["response"].iter_lines(chunk_size=10):
        if line:
            line = line.decode("utf-8")
            print("STREAM LINE:", line)

            # Remove SSE "data: " prefix if present
            if line.startswith("data: "):
                line = line[6:]

            chunks.append(line)
    raw_body = "\n".join(chunks)

# ------------------------------------------------------------
# 8. Fallback: unknown content type (print full response)
# ------------------------------------------------------------
else:
    print("Unknown contentType. Full response:")
    print(response)

print("\nRaw body:")
print(raw_body)

# ------------------------------------------------------------
# 9. Attempt to parse JSON output
# ------------------------------------------------------------
try:
    data = json.loads(raw_body)
    # Adjust the key according to your container's response schema
    # e.g., "response", "output", etc.
    print("\nParsed JSON:")
    print(data)
except json.JSONDecodeError:
    print(
        "\n(The response is not valid JSON. "
        "Check the container logs for debugging.)"
    )
