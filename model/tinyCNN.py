import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyCNN, self).__init__()
        # 32x32x3 -> 16 filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # output 16x32x32
        self.pool = nn.MaxPool2d(2, 2)                           # output 16x16x16

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # output 32x16x16
        # pooling -> 32x8x8

        # fully connected
        self.fc1 = nn.Linear(32*8*8, 128)  # ~32*64 = 2048 features
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (16,16,16)
        x = self.pool(F.relu(self.conv2(x)))   # (32,8,8)
        x = x.view(-1, 32*8*8)                 # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    model =  models.resnet18()
    print(len(model.state_dict()))
    for name, tensor in model.state_dict().items():
        print(f"{name}: {tensor.shape}")

