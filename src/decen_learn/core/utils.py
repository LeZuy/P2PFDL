"""Utility helpers for manipulating model weight dictionaries."""

from typing import Dict, Optional

import torch


def flatten_weight_dict(
    weights: Dict[str, torch.Tensor],
    *,
    detach: bool = False,
    to_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Flatten a dictionary of tensors into a single 1D tensor."""
    flat_parts = []
    target_device = to_device
    for tensor in weights.values():
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        if detach:
            tensor = tensor.detach()
        if target_device is not None:
            tensor = tensor.to(target_device)
        flat_parts.append(tensor.reshape(-1))
    if not flat_parts:
        device = target_device or torch.device("cpu")
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(flat_parts)


def unflatten_weight_dict(
    flat: torch.Tensor,
    template: Dict[str, torch.Tensor],
    *,
    clone: bool = True,
    to_device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct a weight dictionary from flattened tensor using a template."""
    result: Dict[str, torch.Tensor] = {}
    pos = 0
    for name, arr in template.items():
        size = arr.numel()
        slice_tensor = flat[pos:pos + size].reshape_as(arr)
        if to_device is not None:
            slice_tensor = slice_tensor.to(to_device)
        result[name] = slice_tensor.clone() if clone else slice_tensor
        pos += size
    return result
