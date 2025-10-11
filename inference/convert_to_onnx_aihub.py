import os

import torch
import torch.nn as nn


# --- 1. 모델 구조 정의 (train_aihub.py에서 SmallCNN 클래스 복사) ---
class SmallCNN(nn.Module):
    """
    train_aihub.py에서 사용된 SmallCNN 모델 구조
    """
    def __init__(self, n_mels: int = 128, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Global Average Pooling (GAP) 후 Linear
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        h = self.features(x)
        h = h.mean(dim=[2, 3])         # GAP → [B, 128]
        logits = self.classifier(h)    # [B, 3]
        return logits

# --- 2. 설정 및 경로 ---
# train_aihub.py의 Config 값 사용
N_MELS = 128
N_CLASSES = 3
MAX_FRAMES = 64  # 입력 멜스펙의 시간 프레임 길이

# 학습된 모델 파일 경로
MODEL_PATH = 'weights/2025-09-23_03-51_aihub/model_aihub.pth'
# ONNX 출력 파일 경로
ONNX_PATH = 'weights/model_aihub.onnx'


def convert_to_onnx():
    print(f"모델 로딩: {MODEL_PATH}")

    # 1. 모델 인스턴스 생성
    model = SmallCNN(n_mels=N_MELS, n_classes=N_CLASSES)

    # 2. state_dict 로드 (저장 포맷에서 'model' 키를 찾아 상태를 로드)
    try:
        # Pytorch에서 저장한 딕셔너리를 로드
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # 'model' 키에 저장된 state_dict를 사용하여 모델 상태 로드
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("모델 상태(state_dict) 로드 완료.")
        else:
            # 만약 state_dict만 직접 저장했다면 이 방식 사용
            model.load_state_dict(checkpoint)
            print("모델 상태(state_dict) 로드 완료 (직접 저장된 state_dict).")

    except FileNotFoundError:
        print(f"[오류] 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    except Exception as e:
        print(f"[오류] 모델 로딩 중 문제 발생: {e}")
        return

    # 3. 모델을 평가 모드로 설정
    model.eval()

    # 4. Dummy Input 생성
    # [Batch Size, Channel, Mel Bands, Time Frames] = [1, 1, 128, 64]
    dummy_input = torch.randn(1, 1, N_MELS, MAX_FRAMES)
    print(f"더미 입력 텐서 생성 완료. Shape: {dummy_input.shape}")

    # 5. ONNX로 변환
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    
    # 입력 및 출력 이름을 명시하여 ONNX 런타임에서 사용하기 쉽게 합니다.
    input_names = ["input_mel_spectrogram"]
    output_names = ["output_logits"]
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=13 # 호환성 높은 opset 버전 사용
    )

    print("-" * 30)
    print(f"✅ ONNX 변환 완료!")
    print(f"저장 경로: {ONNX_PATH}")
    print("-" * 30)


if __name__ == "__main__":
    convert_to_onnx()
if __name__ == "__main__":
    convert_to_onnx()
