import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Assuming input image size is 128x128, calculate output size after conv/pool
        # 128 -> 64 (pool1) -> 32 (pool2) -> 16 (pool3)
        self.fc1 = nn.Linear(64 * 16 * 16, 64)  # 64 channels * 16x16 feature map
        self.fc2 = nn.Linear(64, 2)  # 2 output classes: cat and dog

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 64, 16, 16]
        x = x.view(-1, 64 * 16 * 16)           # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
