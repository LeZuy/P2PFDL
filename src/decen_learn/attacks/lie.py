# src/decen_learn/attacks/lie.py
"""Little Is Enough (LIE) attack implementation."""

import math
import numpy as np
import torch
from scipy.stats import norm
from typing import Dict, List, Optional

from .base import BaseAttack


class LIEAttack(BaseAttack):
    """Little Is Enough (LIE) attack.
    
    Crafts adversarial updates based on statistical properties of honest updates.
    The attack computes malicious vectors as: μ + z * σ, where μ and σ are the
    mean and standard deviation of honest updates, and z is chosen to maximize
    impact while avoiding detection.
    
    Reference:
        Baruch et al. "A Little Is Enough: Circumventing Defenses For 
        Distributed Learning" (NeurIPS 2019)
    """
    
    def __init__(
        self,
        boosting_factor: float = 1.0,
        z_max: Optional[float] = None,
        num_byzantine: Optional[int] = None,
        num_total: Optional[int] = None,
    ):
        """Initialize LIE attack.
        
        Args:
            boosting_factor: Additional amplification factor
            z_max: Pre-computed z value. If None, computed from num_byzantine and num_total
            num_byzantine: Number of Byzantine workers (m)
            num_total: Total number of workers (n)
        """
        super().__init__(boosting_factor)
        self.z_max = z_max
        self.num_byzantine = num_byzantine
        self.num_total = num_total
        
        if z_max is None and (num_byzantine is not None and num_total is not None):
            self.z_max = self._compute_z_max(num_total, num_byzantine)
    
    @staticmethod
    def _compute_z_max(n: int, m: int) -> float:
        """Compute optimal z value for LIE attack.
        
        Args:
            n: Total number of workers
            m: Number of Byzantine workers
            
        Returns:
            Optimal z value for standard normal distribution
        """
        s = math.ceil(n / 2 + 1) - m  # Required supporters
        benign = n - m
        
        if benign <= 0:
            raise ValueError("No benign workers (n - m <= 0).")
        
        target = (n - s) / benign
        # Clip numerical edge cases: ensure target in (0,1)
        target = min(max(target, 1e-12), 1 - 1e-12)
        z_max = norm.ppf(target)
        
        return z_max
    
    def craft(
        self,
        honest_weights: List[Dict[str, np.ndarray]],
        attacker_weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Craft malicious weights using LIE strategy.
        
        Args:
            honest_weights: List of weight dicts from honest neighbors
            attacker_weights: The attacker's current weights (unused)
            
        Returns:
            Malicious weight dictionary
        """
        if not honest_weights:
            return attacker_weights.copy()
        
        # Flatten all honest weights
        honest_flat = np.stack([
            self._flatten(w) for w in honest_weights
        ])
        
        # Compute statistics
        mu = honest_flat.mean(axis=0)
        sigma = honest_flat.std(axis=0, ddof=0)
        
        # Compute z_max if not provided
        if self.z_max is None:
            if self.num_byzantine is not None and self.num_total is not None:
                z = self._compute_z_max(self.num_total, self.num_byzantine)
            else:
                # Default: use moderate perturbation
                z = 1.0
        else:
            z = self.z_max
        
        # Craft malicious update: μ + z * σ
        malicious_flat = (mu + z * sigma) * self.boosting_factor
        
        # Unflatten back to dictionary structure
        return self._unflatten(malicious_flat, attacker_weights)


# Helper functions
def required_supporters(n: int, m: int) -> int:
    """Compute required supporters for majority voting."""
    return math.ceil(n / 2 + 1) - m


def compute_z_max(n: int, m: int) -> tuple:
    """Compute optimal z value and related statistics.
    
    Args:
        n: Total workers
        m: Byzantine workers
        
    Returns:
        Tuple of (required_supporters, target_fraction, z_max)
    """
    s = required_supporters(n, m)
    benign = n - m
    
    if benign <= 0:
        raise ValueError("No benign workers (n - m <= 0).")
    
    target = (n - s) / benign
    target = min(max(target, 1e-12), 1 - 1e-12)
    z_max = norm.ppf(target)
    
    return s, target, z_max


# Legacy torch interface
def craft_malicious_vector(
    vectors: torch.Tensor,
    z: float,
    n: int,
    m: int,
) -> torch.Tensor:
    """Craft malicious vector using LIE attack - legacy torch interface.
    
    Args:
        vectors: Tensor of honest vectors (num_honest, d)
        z: Standard score for perturbation
        n: Total number of workers
        m: Number of Byzantine workers
        
    Returns:
        Malicious vector
    """
    mu = vectors.mean(dim=0)
    sigma = vectors.std(dim=0, unbiased=False)
    return mu + z * sigma