from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
import logging

from app.routers import sample
from app.recipe_app import RecipeApp

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

app = FastAPI()

# Create a singleton RecipeApp instance shared across requests
recipe_app = RecipeApp()

# Jinja templates (web UI)
templates = Jinja2Templates(directory="app/templates")

# Mount static assets and include API routers
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(sample.router, prefix="/api")


# ------------------------------------------------------------
# Local/Web chat endpoint
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    chat_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """Internal endpoint for local/Web chat usage."""
    return {"response": await recipe_app.chat(req.chat_id, req.message)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the interactive web chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------------------------------------------
# AgentCore-required endpoints
# ------------------------------------------------------------
@app.get("/ping")
def ping():
    """Health check endpoint for AgentCore Runtime."""
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(req: Request):
    """
    Unified invocation entrypoint for AgentCore Runtime.

    AgentCore wraps model inputs (e.g., "prompt") and sends JSON to this endpoint.
    Expected schema:
      body = {"input": {"prompt": "..."}}
    """
    body = await req.json()
    logging.info("[AgentCore] Received request payload: %s", body)

    prompt = (body.get("input") or {}).get("prompt", "")

    # Reuse the same chat logic as the local/Web endpoint.
    response_text = await recipe_app.chat(
        chat_id="agentcore_session",
        message=prompt,
    )

    # Standardize the response schema for AgentCore.
    return JSONResponse(content={"output": response_text})
