from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os, sys, tempfile
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.predict_tags_occx import TagPredictorONNX

router = APIRouter(prefix="/ai")

class VoiceAnalysisRequest(BaseModel):
    s3_uri: str

class VoiceTypeResult(BaseModel):
    type: str
    ratio: float

class VoiceAnalysisData(BaseModel):
    top_voice_types: List[VoiceTypeResult]

class VoiceAnalysisResponse(BaseModel):
    status: int
    message: str
    data: VoiceAnalysisData

@router.post("/voice-analysis", response_model=VoiceAnalysisResponse)
async def analyze_voice(request: VoiceAnalysisRequest):
    LABEL_MAPPING = {
        'belt': 0, 'breathy': 1, 'fast_forte': 2, 'fast_piano': 3,
        'forte': 4, 'inhaled': 5, 'lip_trill': 6, 'messa': 7,
        'pp': 8, 'slow_forte': 9, 'slow_piano': 10, 'spoken': 11,
        'straight': 12, 'trill': 13, 'vibrato': 14, 'vocal_fry': 15
    }

    model_path = "weights/model.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {model_path}")

    try:
        response = requests.get(request.s3_uri)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"S3 파일 요청 실패: {str(e)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        predictor = TagPredictorONNX(model_path=model_path, label_mapping=LABEL_MAPPING)
        result = predictor.predict(tmp_file_path)

        result_objs = [
            VoiceTypeResult(type=tag, ratio=round(prob, 4))
            for tag, prob in result
        ]

        return VoiceAnalysisResponse(
            status=200,
            message="발성 분석이 완료되었습니다.",
            data=VoiceAnalysisData(top_voice_types=result_objs)
        )
    finally:
        os.remove(tmp_file_path)
