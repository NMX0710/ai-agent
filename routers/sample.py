from fastapi import APIRouter

router = APIRouter(prefix="/sample", tags=["sample"])

@router.get("/")
def sample():
    return {"message": "这是一个测试接口"}
