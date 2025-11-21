# src/decen_learn/attacks/minmax.py
"""MinMax attack implementation."""

from typing import Dict, List, Callable, Tuple, Optional
import numpy as np
import torch

from .base import BaseAttack


MIN_GAMMA = 1e-6


class MinMaxAttack(BaseAttack):
    """MinMax attack that maximizes perturbation while staying within convex hull.
    
    The attack crafts adversarial updates by finding the maximum scaling factor
    (gamma) such that the malicious vector stays within the convex hull of
    honest updates, using either minmax or minsum oracle.
    
    Reference:
        Shejwalkar and Houmansadr. "Manipulating the Byzantine: Optimizing 
        Model Poisoning Attacks and Defenses for Federated Learning" (NDSS 2021)
    """
    
    def __init__(
        self,
        boosting_factor: float = 1.0,
        gamma_init: float = 20.0,
        tau: float = 1e-3,
        max_iter: int = 200,
        oracle_type: str = "minmax",
        perturb_kind: str = "auto",
    ):
        """Initialize MinMax attack.
        
        Args:
            boosting_factor: Additional amplification factor
            gamma_init: Initial gamma value for binary search
            tau: Convergence tolerance for gamma optimization
            max_iter: Maximum iterations for binary search
            oracle_type: Oracle constraint type ("minmax" or "minsum")
            perturb_kind: Perturbation direction ("auto", "unit", "std", "sign")
        """
        super().__init__(boosting_factor)
        self.gamma_init = gamma_init
        self.tau = tau
        self.max_iter = max_iter
        self.oracle_type = oracle_type.lower()
        self.perturb_kind = perturb_kind.lower()
        
        if self.oracle_type not in ("minmax", "minsum"):
            raise ValueError(f"Invalid oracle_type: {oracle_type}")
    
    def craft(
        self,
        honest_weights: List[Dict[str, np.ndarray]],
        attacker_weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Craft malicious weights using MinMax strategy.
        
        Args:
            honest_weights: List of weight dicts from honest neighbors
            attacker_weights: The attacker's current weights
            
        Returns:
            Malicious weight dictionary
        """
        if not honest_weights:
            return attacker_weights.copy()
        
        # Flatten all weights
        honest_flat = np.stack([
            self._flatten(w) for w in honest_weights
        ])
        
        # Compute reference point (mean)
        v_ref = honest_flat.mean(axis=0)
        
        # Compute perturbation direction
        perturb_dir = self._compute_perturbation_direction(
            honest_flat,
            self.perturb_kind
        )
        
        # Select oracle
        if self.oracle_type == "minmax":
            oracle = self._check_minmax
        else:
            oracle = self._check_minsum
        
        # Optimize gamma
        gamma = self._optimize_gamma(
            honest_flat,
            perturb_dir,
            oracle,
            v_ref
        )
        
        # Craft malicious vector
        malicious_flat = v_ref + self.boosting_factor * gamma * perturb_dir
        
        return self._unflatten(malicious_flat, attacker_weights)
    
    def _compute_perturbation_direction(
        self,
        vectors: np.ndarray,
        kind: str
    ) -> np.ndarray:
        """Compute perturbation direction based on strategy."""
        if kind == "auto":
            return self._perturb_inverse_std(vectors)
        elif kind == "unit":
            return self._perturb_inverse_unit(vectors)
        elif kind == "std":
            return self._perturb_inverse_std(vectors)
        elif kind == "sign":
            return self._perturb_inverse_sign(vectors)
        else:
            raise ValueError(f"Unknown perturb_kind: {kind}")
    
    def _perturb_inverse_unit(self, vectors: np.ndarray) -> np.ndarray:
        """Negative normalized mean direction."""
        v_bar = vectors.mean(axis=0)
        norm = np.linalg.norm(v_bar)
        if norm < 1e-12:
            return np.zeros_like(v_bar)
        return -v_bar / norm
    
    def _perturb_inverse_std(self, vectors: np.ndarray) -> np.ndarray:
        """Negative normalized standard deviation direction."""
        std = vectors.std(axis=0, ddof=0)
        if np.allclose(std, 0):
            return self._perturb_inverse_unit(vectors)
        norm = np.linalg.norm(std)
        if norm < 1e-12:
            return np.zeros_like(std)
        return -std / norm
    
    def _perturb_inverse_sign(self, vectors: np.ndarray) -> np.ndarray:
        """Negative normalized sign direction."""
        mu = vectors.mean(axis=0)
        s = -np.sign(mu)
        zero_idx = s == 0
        if zero_idx.any():
            s[zero_idx] = np.random.rand(zero_idx.sum()) - 0.5
        norm = np.linalg.norm(s)
        if norm < 1e-12:
            return np.zeros_like(s)
        return s / norm
    
    def _check_minmax(
        self,
        vectors: np.ndarray,
        v_m: np.ndarray
    ) -> bool:
        """Check minmax oracle constraint."""
        dists = np.linalg.norm(
            vectors[:, None, :] - vectors[None, :, :],
            axis=2
        )
        R_max = float(np.max(dists))
        dist_to_m = np.linalg.norm(vectors - v_m[None, :], axis=1)
        max_dist = float(np.max(dist_to_m))
        return max_dist <= (R_max + 1e-12)
    
    def _check_minsum(
        self,
        vectors: np.ndarray,
        v_m: np.ndarray
    ) -> bool:
        """Check minsum oracle constraint."""
        dists = np.linalg.norm(
            vectors[:, None, :] - vectors[None, :, :],
            axis=2
        )
        S_max = float(np.max(np.sum(dists ** 2, axis=1)))
        sum_m = float(np.sum(np.linalg.norm(vectors - v_m[None, :], axis=1) ** 2))
        return sum_m <= (S_max + 1e-12)
    
    def _optimize_gamma(
        self,
        vectors: np.ndarray,
        perturb_dir: np.ndarray,
        oracle: Callable[[np.ndarray, np.ndarray], bool],
        v_ref: np.ndarray,
    ) -> float:
        """Binary search for maximum valid gamma."""
        g = float(self.gamma_init)
        step = g / 2.0
        gamma_succ = 0.0
        prev_g = None
        
        for _ in range(self.max_iter):
            v_m_candidate = v_ref + g * perturb_dir
            cond = oracle(vectors, v_m_candidate)
            
            if cond:
                gamma_succ = g
                g = g + step
            else:
                g = g - step
            
            if g < MIN_GAMMA:
                g = MIN_GAMMA
            
            step = step / 2.0
            
            if prev_g is not None and abs(g - prev_g) < self.tau:
                break
            prev_g = g
        
        return float(gamma_succ)


# Legacy torch interface
@torch.no_grad()
def craft_malicious_vector(
    vectors: torch.Tensor,
    consensus_type: str,
    oracle_type: str,
    gamma_init: float = 10.0,
    tau: float = 1e-3,
    perturb_kind: str = "auto",
    max_iter: int = 200,
) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
    """Craft malicious vector using MinMax - legacy torch interface."""
    device = vectors.device
    v_ref = vectors.mean(dim=0)
    
    # Compute perturbation direction
    def safe_normalize(v):
        norm = torch.norm(v, p=2)
        return v / norm if norm > 1e-12 else torch.zeros_like(v)
    
    if perturb_kind == "auto" or perturb_kind == "std":
        std = torch.std(vectors, dim=0, unbiased=False)
        perturb_dir = -safe_normalize(std) if not torch.allclose(std, torch.zeros_like(std)) else -safe_normalize(v_ref)
    elif perturb_kind == "unit":
        perturb_dir = -safe_normalize(v_ref)
    else:
        perturb_dir = -safe_normalize(v_ref)
    
    # Select oracle
    if oracle_type.lower() == "minmax":
        def oracle(vecs, v_m):
            D = torch.cdist(vecs, vecs, p=2)
            R_max = float(torch.max(D))
            dist_to_m = torch.norm(vecs - v_m.unsqueeze(0), dim=1)
            return float(torch.max(dist_to_m)) <= (R_max + 1e-12)
    else:
        def oracle(vecs, v_m):
            D = torch.cdist(vecs, vecs, p=2)
            S_max = float(torch.max(torch.sum(D ** 2, dim=1)))
            sum_m = float(torch.sum(torch.norm(vecs - v_m.unsqueeze(0), dim=1) ** 2))
            return sum_m <= (S_max + 1e-12)
    
    # Binary search
    g = float(gamma_init)
    step = g / 2.0
    gamma_succ = 0.0
    prev_g = None
    
    for _ in range(max_iter):
        v_m_candidate = v_ref + g * perturb_dir
        cond = oracle(vectors, v_m_candidate)
        
        if cond:
            gamma_succ = g
            g = g + step
        else:
            g = g - step
        
        if g < MIN_GAMMA:
            g = MIN_GAMMA
        step = step / 2.0
        
        if prev_g is not None and abs(g - prev_g) < tau:
            break
        prev_g = g
    
    v_m = v_ref + gamma_succ * perturb_dir
    return v_m, gamma_succ, perturb_dir, v_ref