from fastapi import FastAPI
from app.routers import sample
from pydantic import BaseModel
from app.recipe_app import RecipeApp
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
