import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import onnxruntime as ort

# --- 프로젝트 루트 설정 (경로 안정화) ---
# 이 스크립트 파일의 위치를 기준으로 프로젝트 루트를 찾습니다.
try:
    # 현재 스크립트 파일의 상위 두 단계 디렉터리를 프로젝트 루트로 설정
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # __file__이 정의되지 않은 환경(REPL 등)을 위한 대체 경로
    PROJECT_ROOT = Path(os.getcwd()).parent
# ----------------------------------------------------

# --- 전처리 설정 (train_aihub.py의 Config 값 사용) ---
# NOTE: 실제 상용 서비스에서는 F0 피치 트래킹 기반의 정밀한 노트 세그멘테이션이 필요합니다.
# 여기서는 train_aihub.py의 멜스펙트럼 및 고정 청크(1.02초) 분할 방식을 모방합니다.
TARGET_SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 50
FMAX = 8000
MAX_FRAMES = 64  # 입력 멜스펙의 시간 프레임 길이 (64*16ms ≈ 1.02s)

# 예측할 3가지 태그와 인덱스 매핑 (BCEWithLogitsLoss 순서)
TAG_MAP = {'bending': 0, 'vibrt': 1, 'breath': 2}
IDX_TO_TAG = {v: k for k, v in TAG_MAP.items()}

def _process_audio_to_mel_chunks(audio_path: str) -> List[np.ndarray]:
    """
    오디오 파일을 로드하고, 멜 스펙트로그램을 생성한 후, 
    모델 입력 크기에 맞게 고정 길이 청크로 분할합니다.
    """
    # 1. 오디오 로드 및 RMS 정규화
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    y = y - np.mean(y)
    y = y / (np.sqrt(np.mean(y ** 2) + 1e-8))

    # 2. 풀 멜 스펙트로그램 생성 (power=2.0)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max) # [n_mels, T_all]
    
    # 3. 고정 길이 청크로 분할 (overlap 0으로 단순화)
    mel_chunks = []
    hop_frames = MAX_FRAMES
    
    # 오디오가 너무 짧으면 처리하지 않음 (최소 1초 필요)
    if S_db.shape[1] < MAX_FRAMES * 0.5:
        return []

    for i in range(0, S_db.shape[1], hop_frames):
        chunk = S_db[:, i:i + MAX_FRAMES]
        if chunk.shape[1] < MAX_FRAMES:
            # 마지막 청크는 패딩
            pad_total = MAX_FRAMES - chunk.shape[1]
            left = pad_total // 2
            right = pad_total - left
            chunk = np.pad(chunk, ((0, 0), (left, right)), mode="constant")
        
        mel_chunks.append(chunk)

    return mel_chunks


class VocalFeaturePredictorONNX:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        # 이 모델은 Sigmoid(멀티라벨) 출력을 사용하므로 Softmax는 사용하지 않음

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """BCEWithLogitsLoss로 학습된 모델의 출력을 확률로 변환"""
        return 1 / (1 + np.exp(-x))

    def predict_ratios(self, audio_path: str) -> Dict[str, float]:
        """
        - audio_path: .wav 파일 경로
        - 반환: 각 태그별 True 예측 비율 (%)
        """
        try:
            # 1. 오디오 전처리 및 청크 분할
            mel_chunks = _process_audio_to_mel_chunks(audio_path)
            if len(mel_chunks) == 0:
                 raise ValueError("오디오 길이가 너무 짧아 분석할 청크가 없습니다.")

            # 2. 모든 청크에 대해 예측 수행
            all_predictions = []
            for mel in mel_chunks:
                # ONNX 모델 입력 shape에 맞게: (1, 1, 128, 64)
                input_array = np.expand_dims(np.expand_dims(mel, axis=0), axis=0).astype(np.float32)
                ort_inputs = {self.input_name: input_array}
                
                # 결과는 Logits (Sigmoid 이전 값)
                logits = self.session.run(None, ort_inputs)[0].flatten() # shape: (3,)
                probs = self._sigmoid(logits)
                
                # 예측: 확률이 0.5를 넘으면 True (1), 아니면 False (0)
                preds = (probs > 0.5).astype(np.int32)
                all_predictions.append(preds)

            # 3. 결과 집계 (청크별 예측 결과를 누적)
            # all_predictions shape: [N_chunks, 3]
            aggregated_preds = np.stack(all_predictions, axis=0)
            num_chunks = aggregated_preds.shape[0]

            # 4. 태그별 True 비율 계산 (True 예측 개수 / 전체 청크 개수)
            tag_counts = np.sum(aggregated_preds, axis=0) # [bending_count, vibrt_count, breath_count]
            
            result = {}
            for idx, tag in IDX_TO_TAG.items():
                ratio = (tag_counts[idx] / num_chunks) * 100 # 비율을 %로
                result[tag] = round(float(ratio), 2)
            
            return result

        except Exception as e:
            # librosa/onnxruntime 관련 오류는 여기에 잡힘
            raise RuntimeError(f"발성 특징 예측 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # --- 경로 수정: PROJECT_ROOT 사용 ---
    MODEL_FILENAME = "model_aihub.onnx"
    # pathlib를 사용하여 경로를 구성
    model_path = PROJECT_ROOT / "weights" / MODEL_FILENAME 
    # ------------------------------------

    # 이 부분은 테스트 환경에서 실제 WAV 파일 경로로 대체해야 합니다.
    try:
        # test_audio.wav가 PROJECT_ROOT에 있다고 가정하고 경로 설정
        test_audio_path = PROJECT_ROOT / "test_audio.wav" 
        
        # Path 객체의 .exists() 및 문자열 변환
        if model_path.exists():
            if not test_audio_path.exists():
                 print(f"경고: 테스트 오디오 파일 '{test_audio_path}'을 찾을 수 없습니다. 실제 WAV 파일을 준비해주세요.")
            else:
                # VocalFeaturePredictorONNX는 Path 객체를 문자열로 자동 변환하여 받습니다.
                predictor = VocalFeaturePredictorONNX(model_path) 
                result = predictor.predict_ratios(str(test_audio_path))
                print("--- 예측 결과 ---")
                print(result)
        else:
            print(f"경고: 모델 파일 '{model_path}'을 찾을 수 없습니다. 예측을 건너뜁니다.")

    except ImportError:
        print("경고: librosa 또는 onnxruntime이 설치되어 있지 않아 예시를 실행할 수 없습니다.")
