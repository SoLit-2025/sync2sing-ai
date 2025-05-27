import torch
import torch.nn as nn
import torch.nn.functional as F

class AnnotatedVocalSetCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnnotatedVocalSetCNN, self).__init__()
        # 입력: (batch, 1, 128, 258) [Mel Spectrogram]
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # → (32, 64, 129)
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # → (64, 32, 64)
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # → (128, 16, 32)
        )
        
        # Global Average Pooling → (128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
