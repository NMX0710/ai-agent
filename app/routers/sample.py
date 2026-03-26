from fastapi import APIRouter

router = APIRouter(prefix="/sample", tags=["sample"])


@router.get("/")
def sample():
    return {"message": "This is a sample endpoint."}
