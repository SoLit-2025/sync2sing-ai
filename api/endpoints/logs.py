# logs.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/logs")
async def get_logs():
    # TODO: 최근 로그 반환
    return {"logs": []}


@router.delete("/logs")
async def delete_logs():
    # TODO: 로그 초기화
    return {"message": "Logs deleted"}
