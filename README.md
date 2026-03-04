# 🍳 AI Recipe Agent

A **LangGraph + MCP powered AI agent** for recipe recommendation, nutrition reasoning, and tool-augmented cooking assistance.

This project is designed as a **production-oriented agent system** that supports:
- multi-turn memory
- retrieval-augmented generation (RAG)
- dynamic tool calling (including MCP tool servers)
- local development (Web UI / API) and cloud deployment (AWS Bedrock AgentCore)


## Overview

**AI Recipe Agent** combines:
- **LangGraph** workflows for controllable agent execution
- **ReAct-style tool calling** for tool-augmented reasoning
- **RAG** over local Markdown recipe documents
- **MCP (Model Context Protocol)** for stdio-based tool servers
- **FastAPI** runtime with Web UI & API endpoints
- **AWS Bedrock AgentCore** compatible runtime interface

At runtime, the agent can decide whether to:
1. answer directly,
2. retrieve relevant recipe context via RAG,
3. call one or more tools (including MCP tools),
4. use memory to stay consistent across turns,
5. return structured, explainable results.

## Key Features

- 🧠 **Multi-turn memory** via LangGraph Checkpointer
- 🔍 **RAG pipeline** over local Markdown recipe documents
- 🛠️ **Tool-augmented reasoning**
  - Web search
  - Web scraping
  - Terminal command execution
  - Resource download
  - PDF generation
  - File read / write
- 🔗 **MCP integration** (stdio-based tool servers)
- 🌐 **FastAPI Web UI & REST API**
- ☁️ **AWS Bedrock AgentCore compatible runtime**
- 🧪 **Test & debug scripts** for memory, tools, MCP, and end-to-end flows


## How It Works (High-Level Flow)

The system is composed of three layers:

### 1) FastAPI Runtime Layer
- Provides Web UI (Jinja template) and REST API endpoints
- Exposes cloud-friendly endpoints used by AgentCore runtime

### 2) Agent Orchestration Layer (LangGraph)
- Defines the multi-step agent workflow
- Maintains multi-turn memory using a checkpointer
- Runs a ReAct-style agent that can decide when to call tools

### 3) Retrieval + Tools Layer
- **RAG pipeline** retrieves relevant recipe context from local Markdown documents
- **Tool registry** provides callable tools (web, scraping, terminal, pdf, download, file IO)
- **MCP client** discovers and wraps stdio-based MCP tools at runtime (e.g., nutrition / recipe search)


## Project Structure

```text
app/
├── recipe_app.py              # Core agent logic (LangGraph + Tools + RAG)
├── rag/
│   └── recipe_app_rag_pipeline.py
├── tools/
│   ├── tool_registry.py
│   ├── mcp_client_tools.py
│   ├── web_search_tool.py
│   ├── web_scraping_tool.py
│   ├── terminal_operation_tool.py
│   ├── resource_download_tool.py
│   └── pdf_generation_tool.py
├── mcp_servers/
│   └── nutrition_mcp_server.py
├── templates/
│   └── index.html             # Web chat UI
├── static/
│   └── css/
└── routers/
    └── sample.py

tests/
├── memory_test_runner.py
├── tool_test_runner.py
├── test_mcp_server.py
└── test_recipe_app_rag.py

main.py                         # FastAPI entrypoint
requirements.txt
```
## Tooling Capabilities

The agent can dynamically decide to call tools based on user intent and intermediate reasoning steps.

Supported tools include:

- 🔍 **Web Search** – query external information sources
- 🕷️ **Web Scraping** – extract structured data from web pages
- 📥 **Resource Download** – download external files and assets
- 💻 **Terminal Command Execution** – run safe, sandboxed shell commands
- 📄 **PDF Generation** – generate structured PDF reports
- 🧾 **File Read / Write** – persist intermediate artifacts
- 🍽️ **Nutrition & Recipe Search** – via MCP + Spoonacular API

MCP tools are automatically discovered and wrapped as LangChain-compatible tools at runtime.


## Local Setup

### Install Dependencies

```pip install -r requirements.txt```

### Configure Environment Variables

Create a .env file in the project root for all your API keys

## Run Locally

Start the FastAPI server:`uvicorn main:app --reload`

After startup:
- Web UI: http://localhost:8000
- API Endpoint: `POST /chat`
- `/chat` payload now requires:
  - `user_id`: stable system-generated user identifier
  - `chat_id`: conversation identifier
  - `message`: user input

## Local Postgres (Docker Compose)

Start a local Postgres instance for persistent memory work:

```bash
docker compose up -d postgres
```

Connection details created by default:
- Host: `localhost`
- Port: `5432`
- Database: `ai_agent`
- Username: `ai_agent`
- Password: `ai_agent_dev`

Example app connection string:

```bash
DATABASE_URL=postgresql://ai_agent:ai_agent_dev@localhost:5432/ai_agent
```

Long-term memory defaults:
- `LONG_TERM_MEMORY_ENABLED=1`
- `MEMORY_RETENTION_DAYS=90`
- `MEMORY_MAX_RECORDS_PER_USER=200`
- `MEMORY_RETRIEVE_TOP_K=5`

Stop and remove container:

```bash
docker compose down
```


## AWS Bedrock AgentCore Deployment

This project is compatible with **AWS Bedrock AgentCore Runtime**.

Required Endpoints

- GET /ping – health check endpoint

- POST /invocations – unified agent runtime entrypoint

Example Invocation (boto3):
```
client.invoke_agent_runtime(
    agentRuntimeArn=RUNTIME_ARN,
    runtimeSessionId=str(uuid.uuid4()),
    payload=json.dumps({
        "input": {"prompt": "Plan a high-protein dinner"}
    }).encode("utf-8"),
)
```
