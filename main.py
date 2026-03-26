import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.recipe_app import RecipeApp
from app.routers import sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

app = FastAPI()
recipe_app = RecipeApp()
app.include_router(sample.router, prefix="/api")


class ChatRequest(BaseModel):
    chat_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    return {"response": await recipe_app.chat(req.chat_id, req.message)}
