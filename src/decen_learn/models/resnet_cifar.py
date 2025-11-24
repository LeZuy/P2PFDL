# src/decen_learn/models/resnet_cifar.py
"""ResNet18 adapted for CIFAR-10/100 (32x32 images)."""

import torch
import torch.nn as nn
import torchvision.models as models

import functools

class ResNet18_CIFAR(nn.Module):
    """ResNet18 architecture adapted for CIFAR datasets.
    
    Key modifications from standard ImageNet ResNet18:
    - Smaller initial conv (3x3 instead of 7x7)
    - Stride 1 in first conv (no downsampling)
    - Remove max pooling layer
    - Adjust final FC layer for 10 classes
    
    These changes are necessary because CIFAR images are 32x32
    instead of ImageNet's 224x224.
    """
    
    def __init__(self, num_classes: int = 10):
        """Initialize ResNet18 for CIFAR.
        
        Args:
            num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        """
        super().__init__()
        
        # Start with pretrained=False to avoid ImageNet weights
        self.model = models.resnet18(weights=None,
                                    norm_layer=functools.partial(
                                        nn.BatchNorm2d, 
                                        track_running_stats=False),
                                    )
        
        # Adapt first conv layer for 32x32 input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # CIFAR:    Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.model.conv1 = nn.Conv2d(
            3, 64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        # Remove max pooling (would downsample too aggressively for 32x32)
        self.model.maxpool = nn.Identity()
        
        # Adjust final fully-connected layer for num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        num_params = self.get_num_parameters()
        return f"ResNet18_CIFAR(num_params={num_params:,})"


def create_resnet18_cifar(num_classes: int = 10, pretrained: bool = False) -> ResNet18_CIFAR:
    """Factory function for creating ResNet18_CIFAR.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (not implemented)
        
    Returns:
        ResNet18_CIFAR model
    """
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for CIFAR variant")
    
    return ResNet18_CIFAR(num_classes=num_classes)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = ResNet18_CIFAR(num_classes=10)
    print(model)
    
    # Test with dummy input
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Print model statistics
    total_params = model.get_num_parameters()
    print(f"Total parameters: {total_params:,}")