# src/decen_learn/data/poisoning.py
"""Data poisoning utilities for backdoor attacks."""

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


def add_pixel_pattern(
    params: dict,
    image: torch.Tensor,
    adversarial_index: int = -1,
) -> torch.Tensor:
    """Add pixel-based trigger pattern to an image.
    
    Args:
        params: Dictionary containing poisoning configuration:
            - 'trigger_num': Number of trigger patterns
            - '{i}_poison_pattern': List of (row, col) positions for pattern i
            - 'trigger_value': Pixel value to set (default: 1.0)
        image: Image tensor of shape (C, H, W)
        adversarial_index: Which pattern to use (-1 for all patterns)
        
    Returns:
        Image with trigger pattern applied
    """
    if not torch.is_tensor(image):
        raise ValueError("image must be a torch.Tensor (C, H, W)")
    
    poisoned = image.clone()
    C, H, W = poisoned.shape
    trigger_value = float(params.get('trigger_value', 1.0))
    
    # Collect poison patterns
    poison_patterns = []
    
    if adversarial_index == -1:
        # Use all patterns
        num_triggers = int(params.get('trigger_num', 0))
        for i in range(num_triggers):
            pattern = params.get(f"{i}_poison_pattern", [])
            if pattern:
                poison_patterns.extend(pattern)
    else:
        # Use specific pattern
        pattern = params.get(f"{adversarial_index}_poison_pattern", [])
        if pattern:
            poison_patterns = list(pattern)
    
    if not poison_patterns:
        return poisoned
    
    # Extract valid coordinates
    coords = []
    for pos in poison_patterns:
        try:
            r, c = int(pos[0]), int(pos[1])
            if 0 <= r < H and 0 <= c < W:
                coords.append((r, c))
        except (IndexError, TypeError, ValueError):
            continue
    
    if not coords:
        return poisoned
    
    # Apply trigger to all channels (for color images)
    for r, c in coords:
        if C >= 3:
            poisoned[:, r, c] = trigger_value
        else:
            poisoned[0, r, c] = trigger_value
    
    return poisoned


class PoisonedDataset(Dataset):
    """Dataset wrapper that applies poisoning to samples.
    
    This wraps an existing dataset and applies trigger patterns
    to all samples, changing their labels to the target label.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        params: dict,
        adversarial_index: int = -1,
    ):
        """Initialize poisoned dataset.
        
        Args:
            base_dataset: Underlying clean dataset
            params: Poisoning configuration (see add_pixel_pattern)
            adversarial_index: Which poison pattern to use
        """
        self.base = base_dataset
        self.params = params
        self.adversarial_index = adversarial_index
        self.poison_label = params['poison_label_swap']
    
    def __len__(self) -> int:
        return len(self.base)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get poisoned sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (poisoned_image, poison_label)
        """
        image, _ = self.base[idx]
        
        poisoned_image = add_pixel_pattern(
            self.params,
            image,
            self.adversarial_index
        )
        
        poisoned_label = torch.tensor(
            self.poison_label,
            dtype=torch.long
        )
        
        return poisoned_image, poisoned_label


def get_poison_batch(
    params: dict,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    adversarial_index: int = -1,
    evaluation: bool = False,
    dpr: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Poison a subset of samples in a batch.
    
    Args:
        params: Poisoning configuration containing:
            - 'poisoning_per_batch': Number of samples to poison (optional)
            - 'DPR': Data poisoning rate (fraction, used if per_batch not set)
            - 'poison_label_swap': Target label for poisoned samples
            - 'poison_random': Whether to randomly select samples
            - 'poison_seed': Random seed for reproducibility
        batch: Tuple of (images, labels)
        device: Device to move tensors to
        adversarial_index: Which poison pattern to use
        evaluation: If True, poison entire batch
        dpr: Optional override for data poisoning rate
        
    Returns:
        Tuple of (poisoned_images, poisoned_labels, poison_count)
    """
    images, targets = batch
    
    if images is None or targets is None:
        raise ValueError("Images and targets must be provided")
    
    # Clone and move to device
    new_images = images.clone().to(device)
    new_targets = targets.clone().to(device).long()
    
    batch_size = new_images.size(0)
    if batch_size == 0:
        return new_images, new_targets, 0
    
    # Determine how many samples to poison
    if evaluation:
        # Poison everything in evaluation mode
        num_to_poison = batch_size
    else:
        # Check for explicit count first
        if 'poisoning_per_batch' in params:
            num_to_poison = int(params['poisoning_per_batch'])
        else:
            # Use DPR (data poisoning rate)
            if dpr is None:
                dpr = float(params.get('DPR', 0.0))
            else:
                dpr = float(dpr)
            num_to_poison = int(round(dpr * batch_size))
        
        num_to_poison = max(0, min(num_to_poison, batch_size))
    
    if num_to_poison == 0:
        return new_images, new_targets, 0
    
    # Select indices to poison
    poison_random = bool(params.get('poison_random', True))
    
    if poison_random:
        seed = params.get('poison_seed', None)
        if seed is not None:
            rng = np.random.default_rng(seed)
            selected = rng.choice(batch_size, size=num_to_poison, replace=False)
            indices = torch.tensor(selected, dtype=torch.long, device=device)
        else:
            indices = torch.randperm(batch_size, device=device)[:num_to_poison]
    else:
        # Take first num_to_poison samples
        indices = torch.arange(num_to_poison, device=device, dtype=torch.long)
    
    # Apply poisoning
    poison_label = params['poison_label_swap']
    poison_count = 0
    
    for idx in indices.tolist():
        new_targets[idx] = poison_label
        new_images[idx] = add_pixel_pattern(
            params,
            new_images[idx],
            adversarial_index
        )
        poison_count += 1
    
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    
    return new_images, new_targets, poison_count


if __name__ == "__main__":
    # Test poisoning utilities
    import yaml
    
    # Load example params
    try:
        with open('./configs/params.yaml', 'r') as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        # Create dummy params for testing
        params = {
            'trigger_num': 1,
            '0_poison_pattern': [[0, 0], [0, 1], [1, 0], [1, 1]],
            'trigger_value': 1.0,
            'poison_label_swap': 0,
            'DPR': 0.1,
            'poison_random': True,
        }
    
    print("Testing pixel pattern...")
    image = torch.randn(3, 32, 32)
    poisoned = add_pixel_pattern(params, image, adversarial_index=0)
    
    print(f"Original image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Poisoned image range: [{poisoned.min():.3f}, {poisoned.max():.3f}]")
    print(f"Trigger pixels set: {(poisoned == 1.0).sum().item()}")
    
    print("\nTesting batch poisoning...")
    batch_images = torch.randn(8, 3, 32, 32)
    batch_labels = torch.randint(0, 10, (8,))
    
    poisoned_images, poisoned_labels, count = get_poison_batch(
        params,
        (batch_images, batch_labels),
        device=torch.device('cpu'),
        adversarial_index=0,
    )
    
    print(f"Batch size: {batch_images.size(0)}")
    print(f"Poisoned count: {count}")
    print(f"Target label appears {(poisoned_labels == params['poison_label_swap']).sum().item()} times")