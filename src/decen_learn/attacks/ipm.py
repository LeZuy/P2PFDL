# src/decen_learn/attacks/ipm.py
"""Inner Product Manipulation (IPM) attack implementation."""

from typing import Dict, List, Optional, Sequence

import torch

from .base import BaseAttack


class IPMAttack(BaseAttack):
    """Inner Product Manipulation attack.
    
    Crafts adversarial updates by computing the negative of the honest mean
    with controllable magnitude via epsilon parameter.
    
    Reference:
        Xie et al. "Fall of Empires: Breaking Byzantine-tolerant SGD by 
        Inner Product Manipulation" (UAI 2020)
    """
    
    def __init__(
        self,
        boosting_factor: float = 1.0,
        eps: float = 0.5,
    ):
        """Initialize IPM attack.
        
        Args:
            boosting_factor: Additional amplification factor
            eps: Attack strength controlling magnitude of crafted update
        """
        super().__init__(boosting_factor)
        self.eps = eps
    
    def craft(
        self,
        honest_weights: List[Dict[str, torch.Tensor]],
        attacker_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Craft malicious weights using IPM strategy.
        
        Args:
            honest_weights: List of weight dicts from honest neighbors
            attacker_weights: The attacker's current weights (unused)
            
        Returns:
            Malicious weight dictionary
        """
        if not honest_weights:
            return self._clone_template(attacker_weights)
        
        # Flatten all honest weights
        honest_flat = torch.stack(
            [self._flatten(w) for w in honest_weights],
            dim=0,
        )
        
        # Compute honest mean
        honest_mean = honest_flat.mean(dim=0)
        
        # Craft malicious update: -eps * honest_mean
        malicious_flat = -self.eps * honest_mean * self.boosting_factor
        
        # Unflatten back to dictionary structure
        return self._unflatten(malicious_flat, attacker_weights)


# Legacy torch interface
def craft_ipm_local(
    vectors: torch.Tensor,
    *,
    good_mask: Sequence[bool],
    eps: float = 0.5,
) -> torch.Tensor:
    """IPM attack (local neighbourhood version) - legacy torch interface.

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
        return vectors.mean(dim=0)

    good_mean = vectors[mask_tensor].mean(dim=0)
    theta_bad = -eps * good_mean
    return theta_bad
