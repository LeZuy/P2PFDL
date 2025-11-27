# src/decen_learn/models/__init__.py
"""Model architectures for decentralized learning."""

from .resnet_cifar import ResNet, ResNet20
from .tiny_cnn import TinyCNN

__all__ = [
    "ResNet",
    "ResNet20",
    "TinyCNN",
]
