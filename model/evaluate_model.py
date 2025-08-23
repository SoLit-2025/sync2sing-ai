import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.cnn_model import AnnotatedVocalSetCNN
from model.dataloader import AnnotatedVocalSetDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터셋 및 데이터로더 초기화
    dataset = AnnotatedVocalSetDataset(
        spectrogram_dir="data/spectrograms",
        annotation_dir="annotations/extended_1/with_file_header",
        task_type="technique"
    )
    test_loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    # 모델 로드
    model = AnnotatedVocalSetCNN(num_classes=16).to(device)
    model.load_state_dict(
        torch.load("weights/2025-06-11_17-22/best_model.pth", 
                   map_location=device, 
                   weights_only=True)  # 보안 설정 강화
    )
    model.eval()

    # 실제/예측 레이블 수집
    y_true = []
    y_pred = []

    with torch.no_grad():
        for mel, labels, _ in test_loader:
            mel = mel.to(device)
            outputs = model(mel)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # 혼동 행렬 생성 및 저장
    LABELS = list(dataset.label_to_idx.keys())
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20,15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap='Blues' )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    print("혼동 행렬이 성공적으로 저장되었습니다.")
