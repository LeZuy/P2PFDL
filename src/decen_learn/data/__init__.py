# src/decen_learn/data/__init__.py
"""Data loading and partitioning utilities."""

from .loaders import (
    get_trainloader,
    get_testloader,
    get_poison_testloader,
    create_dataloaders,
)
from .partitioning import (
    partition_data_dirichlet,
    partition_data_iid,
    save_partitioned_data,
)
from .poisoning import (
    PoisonedDataset,
    add_pixel_pattern,
    get_poison_batch,
)

__all__ = [
    # Loaders
    "get_trainloader",
    "get_testloader",
    "get_poison_testloader",
    "create_dataloaders",
    # Partitioning
    "partition_data_dirichlet",
    "partition_data_iid",
    "save_partitioned_data",
    # Poisoning
    "PoisonedDataset",
    "add_pixel_pattern",
    "get_poison_batch",
]