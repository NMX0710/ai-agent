from fastapi import FastAPI, Request
from app.routers import sample
from pydantic import BaseModel
from app.recipe_app import RecipeApp
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import logging

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

app = FastAPI()
recipe_app = RecipeApp()
templates = Jinja2Templates(directory="app/templates")

# 静态文件和路由挂载
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(sample.router, prefix="/api")


# === 原有聊天接口 ===
class ChatRequest(BaseModel):
    chat_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """内部接口：用于本地 / Web 聊天"""
    return {"response": await recipe_app.chat(req.chat_id, req.message)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """渲染交互式 Web 聊天界面"""
    return templates.TemplateResponse("index.html", {"request": request})


# === 新增：AgentCore 要求的健康检查端点 ===
@app.get("/ping")
def ping():
    """AgentCore Runtime 用于健康检查"""
    return {"status": "ok"}


# === 新增：AgentCore 要求的主调用端点 ===
@app.post("/invocations")
async def invocations(req: Request):
    """
    AgentCore Runtime 的统一调用入口。
    它会把模型输入（例如 prompt）包装成 JSON 发到这里。
    """
    body = await req.json()
    logging.info(f"[AgentCore] 收到请求: {body}")

    # 兼容官方 AgentCore 输入格式：
    # body = {"input": {"prompt": "..."}}
    prompt = (body.get("input") or {}).get("prompt", "")

    # 你可以直接复用你的聊天逻辑
    response_text = await recipe_app.chat(chat_id="agentcore_session", message=prompt)

    # 统一返回结构
    return JSONResponse(content={"output": response_text})
