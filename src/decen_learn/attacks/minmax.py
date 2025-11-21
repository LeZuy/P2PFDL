# src/decen_learn/attacks/minmax.py
from src.decen_learning.attacks.attack import BaseAttack
from typing import Dict, List
import numpy as np

class MinMaxAttack(BaseAttack):
    """
    MinMax attack that maximizes perturbation while staying 
    within the convex hull of honest updates.
    """
    
    def __init__(
        self,
        boosting_factor: float = 1.0,
        gamma_init: float = 20.0,
        tau: float = 1e-3,
        max_iter: int = 200,
    ):
        super().__init__(boosting_factor)
        self.gamma_init = gamma_init
        self.tau = tau
        self.max_iter = max_iter
    
    def craft(
        self,
        honest_weights: List[Dict[str, np.ndarray]],
        attacker_weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        # Flatten all weights
        honest_flat = np.stack([
            self._flatten(w) for w in honest_weights
        ])
        
        # Compute attack direction and magnitude
        v_ref = honest_flat.mean(axis=0)
        perturb_dir = self._compute_perturbation_direction(honest_flat)
        gamma = self._optimize_gamma(honest_flat, perturb_dir)
        
        malicious_flat = v_ref + self.boosting_factor * gamma * perturb_dir
        
        return self._unflatten(malicious_flat, attacker_weights)
    
    def _compute_perturbation_direction(self, vectors: np.ndarray) -> np.ndarray:
        """Compute optimal perturbation direction."""
        v_bar = vectors.mean(axis=0)
        direction = -v_bar / (np.linalg.norm(v_bar) + 1e-12)
        return direction
    
    def _optimize_gamma(self, vectors: np.ndarray, direction: np.ndarray) -> float:
        """Binary search for maximum valid gamma."""
        # ... implementation
        pass