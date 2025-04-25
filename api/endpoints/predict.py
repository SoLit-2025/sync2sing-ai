# predict.py
from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
async def predict():
    # TODO: 단일 오디오 분석 및 태그 예측
    return {"message": "Predict endpoint"}
