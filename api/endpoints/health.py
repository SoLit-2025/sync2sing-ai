# health.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    # TODO: 서버 및 모델 상태 확인
    return {"status": "ok"}
