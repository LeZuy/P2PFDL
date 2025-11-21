# src/decen_learn/models/tiny_cnn.py
"""Lightweight CNN for quick experiments on CIFAR."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    """Tiny CNN architecture for CIFAR datasets.
    
    Architecture:
    - Conv1: 3 -> 16 channels, 3x3 kernel, ReLU, MaxPool
    - Conv2: 16 -> 32 channels, 3x3 kernel, ReLU, MaxPool
    - FC1: 32*8*8 -> 128, ReLU
    - FC2: 128 -> num_classes
    
    This is a lightweight model useful for:
    - Quick prototyping
    - Testing federated learning algorithms
    - Low-resource environments
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """Initialize TinyCNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Convolutional layers
        # Input: (batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        # After conv1 + pool: (batch, 16, 16, 16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After conv2 + pool: (batch, 32, 8, 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, 32, 32)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)           # (B, 16, 32, 32)
        x = F.relu(x)
        x = self.pool(x)            # (B, 16, 16, 16)
        
        # Conv block 2
        x = self.conv2(x)           # (B, 32, 16, 16)
        x = F.relu(x)
        x = self.pool(x)            # (B, 32, 8, 8)
        
        # Flatten
        x = x.view(-1, 32 * 8 * 8)  # (B, 2048)
        
        # FC layers
        x = self.fc1(x)             # (B, 128)
        x = F.relu(x)
        x = self.fc2(x)             # (B, num_classes)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        num_params = self.get_num_parameters()
        return (
            f"TinyCNN(num_classes={self.num_classes}, "
            f"input_channels={self.input_channels}, "
            f"num_params={num_params:,})"
        )


def create_tiny_cnn(num_classes: int = 10, input_channels: int = 3) -> TinyCNN:
    """Factory function for creating TinyCNN.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        
    Returns:
        TinyCNN model
    """
    return TinyCNN(num_classes=num_classes, input_channels=input_channels)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = TinyCNN(num_classes=10)
    print(model)
    
    # Test with dummy input
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Print model statistics
    total_params = model.get_num_parameters()
    print(f"Total parameters: {total_params:,}")
    
    # Compare with ResNet18
    print("\nParameter comparison:")
    print(f"TinyCNN: {total_params:,} params (~{total_params/1e6:.2f}M)")
    
    from .resnet_cifar import ResNet18_CIFAR
    resnet = ResNet18_CIFAR(num_classes=10)
    resnet_params = resnet.get_num_parameters()
    print(f"ResNet18: {resnet_params:,} params (~{resnet_params/1e6:.2f}M)")
    print(f"Size ratio: {resnet_params / total_params:.1f}x")