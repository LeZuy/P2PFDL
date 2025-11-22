# src/decen_learn/models/__init__.py
"""Model architectures for decentralized learning."""

from .resnet_cifar import ResNet18_CIFAR, create_resnet18_cifar
from .tiny_cnn import TinyCNN

__all__ = [
    "ResNet18_CIFAR",
    "create_resnet18_cifar",
    "TinyCNN",
]
