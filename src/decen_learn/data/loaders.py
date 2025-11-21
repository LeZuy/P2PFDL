# src/decen_learn/data/loaders.py
"""Data loading utilities for CIFAR datasets."""

from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(
    dataset_name: str = "cifar10",
    train: bool = True,
) -> transforms.Compose:
    """Get appropriate transforms for dataset.
    
    Args:
        dataset_name: Name of dataset ("cifar10", "cifar100", etc.)
        train: Whether to apply training augmentations
        
    Returns:
        Composed transforms
    """
    dataset_name = dataset_name.lower()
    
    # Select normalization constants
    if dataset_name == "cifar10":
        normalize_mean = CIFAR10_MEAN
        normalize_std = CIFAR10_STD
    elif dataset_name == "cifar100":
        normalize_mean = CIFAR100_MEAN
        normalize_std = CIFAR100_STD
    else:
        # Fallback to generic normalization
        normalize_mean = (0.5, 0.5, 0.5)
        normalize_std = (0.5, 0.5, 0.5)
    
    if train:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    else:
        # Test transforms without augmentation
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])


def get_trainloader(
    data_path: str,
    batch_size: int = 64,
    dataset_name: str = "cifar10",
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create training data loader.
    
    Args:
        data_path: Path to training data directory
        batch_size: Batch size
        dataset_name: Dataset name for proper normalization
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader for training
    """
    transform = get_transforms(dataset_name, train=True)
    
    trainset = datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return trainloader


def get_testloader(
    data_path: str,
    batch_size: int = 64,
    dataset_name: str = "cifar10",
    num_workers: int = 0,
) -> DataLoader:
    """Create test data loader.
    
    Args:
        data_path: Path to test data directory
        batch_size: Batch size
        dataset_name: Dataset name for proper normalization
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader for testing
    """
    transform = get_transforms(dataset_name, train=False)
    
    testset = datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return testloader


def get_poison_testloader(
    data_path: str,
    poison_params: dict,
    batch_size: int = 64,
    dataset_name: str = "cifar10",
    num_workers: int = 0,
    adversarial_index: int = -1,
) -> DataLoader:
    """Create poisoned test data loader.
    
    Args:
        data_path: Path to test data directory
        poison_params: Poisoning configuration
        batch_size: Batch size
        dataset_name: Dataset name
        num_workers: Number of workers
        adversarial_index: Which poison pattern to use
        
    Returns:
        DataLoader with poisoned data
    """
    from .poisoning import PoisonedDataset
    
    transform = get_transforms(dataset_name, train=False)
    
    base_testset = datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    
    poisoned_dataset = PoisonedDataset(
        base_testset,
        poison_params,
        adversarial_index=adversarial_index
    )
    
    loader = DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


def create_dataloaders(
    train_path: str,
    test_path: str,
    batch_size: int = 64,
    dataset_name: str = "cifar10",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create both train and test loaders.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        batch_size: Batch size
        dataset_name: Dataset name
        num_workers: Number of workers
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_loader = get_trainloader(
        train_path,
        batch_size=batch_size,
        dataset_name=dataset_name,
        num_workers=num_workers,
    )
    
    test_loader = get_testloader(
        test_path,
        batch_size=batch_size,
        dataset_name=dataset_name,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "./data_splits/client_0"
    
    print(f"Loading data from: {data_path}")
    
    try:
        train_loader = get_trainloader(data_path, batch_size=32)
        print(f"✓ Train loader created: {len(train_loader)} batches")
        
        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Label shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    except Exception as e:
        print(f"✗ Failed to create loader: {e}")