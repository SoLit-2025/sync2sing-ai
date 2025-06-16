import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention 레이어 정의 
class AttentionLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, in_dim)
        weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, seq_len, 1)
        return (x * weights).sum(dim=1)  # (batch_size, in_dim)

# 기존 모델 수정
class AnnotatedVocalSetCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 입력: (batch, 1, 128, 258) [Mel Spectrogram]
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # → (32, 64, 129)
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # → (64, 32, 64)
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # → (128, 16, 32)
        )
        
        # Global Average Pooling 제거 → Attention으로 대체
        self.attention = AttentionLayer(in_dim=128)  # in_dim=128 (채널 수)
        
        # Fully Connected
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv 블록 통과
        x = self.conv_block(x)  # 출력 형태: (batch, 128, 16, 32)
        
        # Attention 입력을 위해 차원 재구성
        # (batch, 128, 16, 32) → (batch, 16*32, 128)
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, 128)
        
        # Attention 적용
        x = self.attention(x)  # 출력 형태: (batch, 128)
        
        # 최종 분류
        return self.fc(x)
