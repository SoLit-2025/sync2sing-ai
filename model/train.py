import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader import AnnotatedVocalSetDataset
from model.cnn_model import AnnotatedVocalSetCNN
import matplotlib.pyplot as plt
import os
import datetime
import torch.multiprocessing
import torch


print(torch.__version__)  # PyTorch 버전 확인
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 (True면 GPU 사용 가능)
print(torch.cuda.get_device_name(0))


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
SAVE_DIR = f"weights/{current_time}"  
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    # 하이퍼파라미터 설정
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 및 데이터로더 초기화
    dataset = AnnotatedVocalSetDataset(
        spectrogram_dir="data/spectrograms",
        annotation_dir="annotations/extended_1/with_file_header",
        task_type="technique"
    )

    # 데이터 분할 (train: 80%, val: 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 모델 초기화
    model = AnnotatedVocalSetCNN(num_classes=dataset.get_num_classes()).to(device)

    # 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 기록 저장
    train_loss_history = []
    val_accuracy_history = []

    # 학습 루프
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training Phase
        for batch_idx, (mel, labels, _) in enumerate(train_loader):
            mel = mel.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(mel)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 통계
            running_loss += loss.item() * mel.size(0)
        
        # Epoch 별 Training Loss 계산
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel, labels, _ in val_loader:
                mel = mel.to(device)
                labels = labels.to(device)
                
                outputs = model(mel)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracy_history.append(val_accuracy)
        
        # 모델 저장 (Best)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        
        # 로그 출력
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")

    # 최종 모델 저장
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    
    return train_loss_history, val_accuracy_history

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    train_loss, val_acc = main()
    
    # 학습 곡선 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
    plt.show()
