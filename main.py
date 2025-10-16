from fastapi import FastAPI, Request
from app.routers import sample
from pydantic import BaseModel
from app.recipe_app import RecipeApp
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 强制输出到终端
)

app = FastAPI()
recipe_app = RecipeApp()
templates = Jinja2Templates(directory="app/templates")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Sample to show how swagger used
app.include_router(sample.router, prefix="/api")

class ChatRequest(BaseModel):
    chat_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    return {"response": await recipe_app.chat(req.chat_id, req.message)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the interactive chat console."""
    return templates.TemplateResponse("index.html", {"request": request})
