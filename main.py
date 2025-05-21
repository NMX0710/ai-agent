from fastapi import FastAPI
from app.routers import sample
from pydantic import BaseModel
from app.recipe_app import RecipeApp

import logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 强制输出到终端
)

app = FastAPI()
recipe_app = RecipeApp()

# Sample to show how swagger used
app.include_router(sample.router, prefix="/api")

class ChatRequest(BaseModel):
    chat_id: str
    message: str
@app.post("/chat")
async def chat(req: ChatRequest):
    return {"response": await recipe_app.chat(req.chat_id, req.message)}
