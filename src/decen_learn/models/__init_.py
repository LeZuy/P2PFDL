# model/__init__.py
"""
DEPRECATED: This module is deprecated and will be removed in a future version.

Please migrate to:
- decen_learn.models (for model architectures)
- decen_learn.data (for data loading, partitioning, poisoning)
- decen_learn.core.LocalTrainer (for training utilities)

See MODEL_MIGRATION.md for detailed migration instructions.
"""

import warnings

warnings.warn(
    "The 'model' module is deprecated. "
    "Use 'decen_learn.models' and 'decen_learn.data' instead. "
    "See MODEL_MIGRATION.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Forward imports for backward compatibility
try:
    from src.decen_learn.models import ResNet18_CIFAR, TinyCNN
    from src.decen_learn.data import (
        get_trainloader,
        get_testloader,
        get_poison_testloader,
        get_poison_batch,
        PoisonedDataset,
        add_pixel_pattern,
    )
    
    __all__ = [
        "ResNet18_CIFAR",
        "TinyCNN",
        "get_trainloader",
        "get_testloader",
        "get_poison_testloader",
        "get_poison_batch",
        "PoisonedDataset",
        "add_pixel_pattern",
    ]
except ImportError as e:
    warnings.warn(
        f"Could not import from new modules: {e}. "
        "Falling back to old imports.",
        ImportWarning
    )