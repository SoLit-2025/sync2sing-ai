from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os, sys, tempfile
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.predict_tags_occx import TagPredictorONNX

try:
    from inference.predict_tags_onnx_aihub import VocalFeaturePredictorONNX
except ImportError:
    # 에러 방지용 더미 클래스
    class VocalFeaturePredictorONNX:
        def __init__(self, model_path): pass
        def predict_ratios(self, audio_path): return {"bending": 0, "vibrt": 0, "breath": 0}

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


class VocalFeaturesResponseData(BaseModel):
    bending: float
    vibrt: float
    breath: float

class VocalFeaturesResponse(BaseModel):
    status: int
    message: str
    data: VocalFeaturesResponseData

@router.post("/voice-analysis-aihub", response_model=VocalFeaturesResponse)
async def get_vocal_features(request: VoiceAnalysisRequest):
    # ONNX 모델 경로 (convert_to_onnx_aihub.py에서 사용한 경로)
    model_path = "weights/model_aihub.onnx"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail=f"AI 모델 파일을 찾을 수 없습니다: {model_path}")

    # 1. S3 URI에서 오디오 파일 다운로드
    try:
        response = requests.get(request.s3_uri, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"S3 파일 요청 실패: {str(e)}")

    tmp_file_path = None
    try:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # 2. 예측 수행
        predictor = VocalFeaturePredictorONNX(model_path=model_path)
        # predict_ratios는 {"bending": 30.0, "vibrt": 40.0, "breath": 50.0} 형태를 반환
        result_ratios = predictor.predict_ratios(tmp_file_path)

        # 3. 응답 모델에 맞게 데이터 구성
        return VocalFeaturesResponse(
            status=200,
            message="발성 특징 분석이 완료되었습니다.",
            data=VocalFeaturesResponseData(
                bending=result_ratios['bending'],
                vibrt=result_ratios['vibrt'],
                breath=result_ratios['breath']
            )
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        # ONNX 예측 실패 (전처리, ONNX 실행 등 오류)
        raise HTTPException(status_code=500, detail=f"AI 예측 시스템 오류: {str(re)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 처리 오류: {str(e)}")
    finally:
        # 임시 파일 정리
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
