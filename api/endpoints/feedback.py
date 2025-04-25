# feedback.py
from fastapi import APIRouter

router = APIRouter()


@router.get("/feedback/templates")
async def get_feedback_templates():
    # TODO: 피드백 템플릿 목록 반환
    return {"templates": []}


@router.post("/feedback/custom")
async def add_custom_template():
    # TODO: 사용자 정의 템플릿 추가
    return {"message": "Custom template added"}
