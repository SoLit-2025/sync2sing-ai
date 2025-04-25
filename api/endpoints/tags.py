# tags.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/tags")
async def get_tags():
    # TODO: 태그 목록 반환
    return {"tags": []}
