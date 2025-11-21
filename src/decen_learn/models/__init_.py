# src/decen_learn/models/__init__.py
"""Model architectures for federated learning experiments."""

from .resnet_cifar import ResNet18_CIFAR
from .tiny_cnn import TinyCNN

__all__ = [
    "ResNet18_CIFAR",
    "TinyCNN",
]