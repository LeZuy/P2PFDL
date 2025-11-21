import torch
from typing import Iterable, Sequence


def craft_ipm_local(
    vectors: torch.Tensor,
    *,
    good_mask: Sequence[bool],
    eps: float = 0.5,
) -> torch.Tensor:
    """
    IPM attack (local neighbourhood version).

    Args:
        vectors: Tensor of shape (N, d) stacked in the same order as good_mask.
        good_mask: Boolean mask (len=N) indicating which entries are honest.
        eps: Attack strength controlling the magnitude of the crafted update.

    Returns:
        A single adversarial vector (d,) tailored against the honest mean.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2-D tensor of shape (N, d)")

    if len(good_mask) != vectors.shape[0]:
        raise ValueError("good_mask must have the same length as vectors")

    mask_tensor = torch.as_tensor(good_mask, dtype=torch.bool, device=vectors.device)
    if not torch.any(mask_tensor):
        # No honest updates available – fall back to zero perturbation.
        return vectors.mean(dim=0)

    good_mean = vectors[mask_tensor].mean(dim=0)
    theta_bad = -eps * good_mean
    return theta_bad
