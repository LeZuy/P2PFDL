# src/decen_learn/data/partitioning.py
"""Data partitioning utilities for federated learning."""

import os
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


def partition_data_dirichlet(
    num_clients: int,
    alpha: float,
    dataset_name: str = "cifar10",
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> Tuple[List[Subset], Dataset]:
    """Partition data according to Dirichlet distribution (non-IID).
    
    The Dirichlet distribution controls the heterogeneity of data distribution:
    - alpha -> 0: More heterogeneous (each client has fewer classes)
    - alpha -> ∞: More homogeneous (approaches IID)
    
    Typical values:
    - alpha = 0.1: Highly non-IID (pathological)
    - alpha = 0.5: Moderately non-IID
    - alpha = 1.0: Mildly non-IID
    - alpha = 10.0: Nearly IID
    
    Args:
        num_clients: Number of clients to partition data among
        alpha: Dirichlet concentration parameter
        dataset_name: Dataset to partition ("cifar10", "cifar100", "mnist")
        seed: Random seed for reproducibility
        min_samples_per_client: Minimum samples each client must have
        
    Returns:
        Tuple of (list of client datasets, test dataset)
    """
    prng = np.random.default_rng(seed)
    
    # Download dataset
    if dataset_name.lower() == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif dataset_name.lower() == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    elif dataset_name.lower() == "mnist":
        trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Get targets
    if hasattr(trainset, 'targets'):
        targets = np.array(trainset.targets)
    elif hasattr(trainset, 'labels'):
        targets = np.array(trainset.labels)
    else:
        # Extract from dataset
        targets = np.array([trainset[i][1] for i in range(len(trainset))])
    
    num_classes = len(np.unique(targets))
    total_samples = len(targets)
    
    # Repeat until minimum samples constraint is met
    min_samples = 0
    max_attempts = 100
    attempt = 0
    
    while min_samples < min_samples_per_client and attempt < max_attempts:
        attempt += 1
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        
        # For each class, distribute samples according to Dirichlet
        for k in range(num_classes):
            # Get indices of samples belonging to class k
            idx_k = np.where(targets == k)[0]
            prng.shuffle(idx_k)
            
            # Sample proportions from Dirichlet distribution
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance: don't assign to clients that already have too much data
            proportions = np.array([
                p * (len(client_idx) < total_samples / num_clients)
                for p, client_idx in zip(proportions, client_indices)
            ])
            proportions = proportions / proportions.sum()
            
            # Split indices according to proportions
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            
            # Assign to clients
            for client_idx, idx_split in zip(client_indices, idx_k_split):
                client_idx.extend(idx_split.tolist())
        
        # Check minimum samples constraint
        min_samples = min(len(idx) for idx in client_indices)
    
    if min_samples < min_samples_per_client:
        raise RuntimeError(
            f"Could not satisfy minimum samples constraint after {max_attempts} attempts. "
            f"Got {min_samples} < {min_samples_per_client}"
        )
    
    # Create Subset datasets for each client
    trainsets_per_client = [
        Subset(trainset, indices) for indices in client_indices
    ]
    
    return trainsets_per_client, testset


def partition_data_iid(
    num_clients: int,
    dataset_name: str = "cifar10",
    seed: int = 42,
) -> Tuple[List[Subset], Dataset]:
    """Partition data uniformly (IID).
    
    Args:
        num_clients: Number of clients
        dataset_name: Dataset to partition
        seed: Random seed
        
    Returns:
        Tuple of (list of client datasets, test dataset)
    """
    prng = np.random.default_rng(seed)
    
    # Download dataset
    if dataset_name.lower() == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Shuffle and split evenly
    indices = np.arange(len(trainset))
    prng.shuffle(indices)
    
    splits = np.array_split(indices, num_clients)
    
    trainsets_per_client = [
        Subset(trainset, split.tolist()) for split in splits
    ]
    
    return trainsets_per_client, testset


def save_partitioned_data(
    client_datasets: List[Dataset],
    testset: Dataset,
    output_dir: str = "./data_splits",
    show_progress: bool = True,
) -> None:
    """Save partitioned datasets as image files.
    
    Saves data in structure:
        output_dir/
            client_0/
                class_0/
                    img_00000.png
                    img_00001.png
                    ...
                class_1/
                    ...
            client_1/
                ...
            test/
                class_0/
                    ...
    
    Args:
        client_datasets: List of client datasets
        testset: Test dataset
        output_dir: Root directory for saved data
        show_progress: Whether to show progress bars
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save training data for each client
    for client_id, dataset in enumerate(client_datasets):
        if show_progress:
            print(f"Saving client {client_id}...")
        
        client_dir = output_path / f"client_{client_id}"
        client_dir.mkdir(exist_ok=True)
        
        iterator = tqdm(range(len(dataset))) if show_progress else range(len(dataset))
        
        for idx in iterator:
            img, label = dataset[idx]
            
            # Create class directory
            class_dir = client_dir / f"class_{label}"
            class_dir.mkdir(exist_ok=True)
            
            # Save image
            img_path = class_dir / f"img_{idx:05d}.png"
            
            if isinstance(img, torch.Tensor):
                # Ensure 3 channels for grayscale images
                if img.ndim == 2:
                    img = img.unsqueeze(0)
                save_image(img, img_path)
            else:
                img.save(img_path)
    
    # Save test data
    if show_progress:
        print("Saving test set...")
    
    test_dir = output_path / "test"
    test_dir.mkdir(exist_ok=True)
    
    iterator = tqdm(range(len(testset))) if show_progress else range(len(testset))
    
    for idx in iterator:
        img, label = testset[idx]
        
        class_dir = test_dir / f"class_{label}"
        class_dir.mkdir(exist_ok=True)
        
        img_path = class_dir / f"img_{idx:05d}.png"
        
        if isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0)
            save_image(img, img_path)
        else:
            img.save(img_path)
    
    print(f"✓ All data saved to {output_dir}")


def analyze_distribution(
    client_datasets: List[Dataset],
    num_classes: int = 10,
) -> dict:
    """Analyze class distribution across clients.
    
    Args:
        client_datasets: List of client datasets
        num_classes: Number of classes in dataset
        
    Returns:
        Dictionary with distribution statistics
    """
    distributions = []
    
    for dataset in client_datasets:
        class_counts = defaultdict(int)
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_counts[label] += 1
        
        # Convert to array
        dist = np.zeros(num_classes)
        for class_id, count in class_counts.items():
            dist[class_id] = count
        
        distributions.append(dist)
    
    distributions = np.array(distributions)
    
    return {
        'distributions': distributions,  # Shape: (num_clients, num_classes)
        'total_samples': distributions.sum(axis=1),
        'classes_per_client': (distributions > 0).sum(axis=1),
        'mean_samples': distributions.sum(axis=1).mean(),
        'std_samples': distributions.sum(axis=1).std(),
    }


if __name__ == "__main__":
    # Test partitioning
    print("Testing Dirichlet partitioning...")
    
    clients, testset = partition_data_dirichlet(
        num_clients=10,
        alpha=0.5,
        dataset_name="cifar10",
        seed=42,
    )
    
    print(f"✓ Created {len(clients)} client datasets")
    print(f"✓ Test set size: {len(testset)}")
    
    # Analyze distribution
    stats = analyze_distribution(clients, num_classes=10)
    
    print("\nDistribution statistics:")
    print(f"  Mean samples per client: {stats['mean_samples']:.1f}")
    print(f"  Std samples per client: {stats['std_samples']:.1f}")
    print(f"  Mean classes per client: {stats['classes_per_client'].mean():.1f}")
    
    print("\nSamples per client:")
    for i, count in enumerate(stats['total_samples']):
        print(f"  Client {i}: {int(count)} samples")