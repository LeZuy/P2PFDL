import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        
        # --- Chỉnh phù hợp cho CIFAR ---
        # Conv đầu nhận ảnh 32×32 thay vì 224×224
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # bỏ maxpool để giữ kích thước
        
        # Lớp fully-connected cuối cho 10 lớp
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)