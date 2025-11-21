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


# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from typing import Tuple, Optional, Callable

# from aggregator import consensus

# MIN_GAMMA = 1e-6

# @torch.no_grad()
# def safe_normalize(v, eps=1e-12):
#     norm = torch.norm(v, p=2)
#     if norm < eps:
#         return torch.zeros_like(v)
#     return v / norm

# @torch.no_grad()
# def perturb_inverse_unit(vectors: torch.Tensor) -> torch.Tensor:
#     """
#     Inverse unit vector: -mean(vectors) normalized.
#     vectors: (n, d)
#     """
#     v_bar = vectors.mean(dim=0)
#     return -safe_normalize(v_bar)

# @torch.no_grad()
# def perturb_inverse_std(vectors: torch.Tensor) -> torch.Tensor:
#     """
#     Inverse std vector: -std(vectors).
#     vectors: (n, d)
#     """
#     std = torch.std(vectors, dim=0, unbiased=False)
#     if torch.allclose(std, torch.zeros_like(std)):
#         return perturb_inverse_unit(vectors)
#     return -safe_normalize(std)


# @torch.no_grad()
# def perturb_inverse_sign(vectors: torch.Tensor) -> torch.Tensor:
#     """
#     Inverse sign vector: -sign(mean(vectors)) normalized.
#     If = 0, add noise to avoid vector 0.
#     vectors: (n, d)
#     """
#     mu = vectors.mean(dim=0)
#     s = -torch.sign(mu)
#     zero_idx = s == 0
#     if zero_idx.any():
#         s[zero_idx] = (torch.rand_like(s[zero_idx]) - 5e-3)
#     return safe_normalize(s)

# def craft_perturb(vectors: torch.Tensor, perturb_kind: str, consensus_type: str) -> torch.Tensor:
#     kind = perturb_kind.lower()
#     if kind == "auto":
#         if consensus_type == "krum":
#             perturb_dir = perturb_inverse_unit(vectors)
#         else:
#             perturb_dir = perturb_inverse_std(vectors)
#     elif kind == "unit":
#         perturb_dir = perturb_inverse_unit(vectors)
#     elif kind == "std":
#         perturb_dir = perturb_inverse_std(vectors)
#     elif kind == "sign":
#         perturb_dir = perturb_inverse_sign(vectors)
#     else:
#         raise ValueError("Unknown perturb_kind: choose 'auto','unit','std','sign'")
#     return perturb_dir

# def check_minmax(vectors: torch.Tensor, v_m: torch.Tensor) -> bool:
#     # R_max = max_{i,j} ||g_i - g_j||
#     # compute pairwise distances (n x n) but only need max
#     # use torch.cdist (n x n) and then max
#     with torch.no_grad():
#         D = torch.cdist(vectors, vectors, p=2)
#         R_max = float(torch.max(D))
#         dist_to_m = torch.norm(vectors - v_m.unsqueeze(0), dim=1)
#         return float(torch.max(dist_to_m)) <= (R_max + 1e-12)

# def check_minsum(vectors: torch.Tensor, v_m: torch.Tensor) -> bool:
#     # S_max = max_i sum_j ||g_i - g_j||^2
#     with torch.no_grad():
#         D = torch.cdist(vectors, vectors, p=2)
#         S_max = float(torch.max(torch.sum(D ** 2, dim=1)))
#         sum_m = float(torch.sum(torch.norm(vectors - v_m.unsqueeze(0), dim=1) ** 2))
#         return sum_m <= (S_max + 1e-12)

# @torch.no_grad()
# def optimize_gamma(vectors: torch.Tensor,
#                    oracle: Callable[[torch.Tensor, torch.Tensor], bool],
#                    perturb_dir: torch.Tensor,
#                    gamma_init: float = 20.0,
#                    tau: float = 1e-3,
#                    max_iter: int = 200,
#                    verbose: bool = False) -> float:

#     g = float(gamma_init)
#     step = g / 2.0
#     gamma_succ = 0.0
#     prev_g = None
   
#     v_ref = vectors.mean(dim=0)

#     for it in range(max_iter):
#         v_m_candidate = v_ref + g * perturb_dir
#         cond = oracle(vectors, v_m_candidate)

#         if cond:
#             gamma_succ = g
#             g = g + step
#         else:
#             g = g - step

#         if g < MIN_GAMMA:
#             g = MIN_GAMMA

#         step = step / 2.0

#         if verbose:
#             print(f"[iter {it}] g={g:.6g}, succ={gamma_succ:.6g}, step={step:.6g}, cond={cond}")

#         if prev_g is not None and abs(g - prev_g) < tau:
#             break
#         prev_g = g

#     return float(gamma_succ)

# @torch.no_grad()
# def craft_malicious_vector(
#     vectors: torch.Tensor,
#     consensus_type: str,
#     oracle_type: str,
#     gamma_init: float = 10.0,
#     tau: float = 1e-3,
#     perturb_kind: str = "auto",
#     max_iter: int = 200,
#     min_gamma: float = 1e-6,
#     agnostic: bool = True,
# ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
#     """
#     Create a malicious vector v_m for AGR-agnostic or AGR-tailored attack.
#     Works for d (dimension).
#     Returns: (v_m, gamma, perturb_dir, v_ref)
#       - v_m: (d,) tensor malicious vector
#       - gamma: float found (0.0 if none)
#       - perturb_dir: (d,) unit L2 direction used
#       - v_ref: (d,) the reference aggregate used (mean or emulated consensus)
#     """

#     n, d = vectors.shape
#     device = vectors.device
#     dtype = vectors.dtype

#     v_ref = vectors.mean(dim=0)
#     v_ref = v_ref.to(device=device, dtype=dtype)

#     perturb_dir = craft_perturb(vectors, "auto", consensus_type)
#     perturb_dir = perturb_dir.to(device=device, dtype=dtype)

#     gamma = optimize_gamma(vectors,  check_minmax, perturb_dir)

#     v_m = v_ref + gamma * perturb_dir

#     return v_m, gamma, perturb_dir, v_ref

# if __name__ == "__main__":
#     torch.manual_seed(0)
#     rule = "krum"
#     vectors = torch.randn(6, 2)

#     v_m, gamma, perturb, v_ref = craft_malicious_vector(
#         vectors=vectors,
#         consensus_type="mean",
#         oracle_type="minmax",
#         gamma_init=20.0,
#         tau=1e-3,
#         perturb_kind="auto",
#     )

#     byzantine_vectors = torch.vstack([v_m] * 3)
#     consensus_vec = consensus(torch.vstack([vectors, byzantine_vectors]), rule)

#     benign_np = vectors.detach().cpu().numpy()
#     byzantine_np = byzantine_vectors.detach().cpu().numpy()
#     consensus_np = consensus_vec.detach().cpu().numpy()
#     ref_np = v_ref.detach().cpu().numpy()

#     plt.figure(figsize=(6, 6))
#     plt.scatter(benign_np[:, 0], benign_np[:, 1], label="Benign vectors", alpha=0.7)
#     plt.scatter(byzantine_np[:, 0], byzantine_np[:, 1], label="Malicious vectors", alpha=0.7)
#     plt.scatter(consensus_np[0], consensus_np[1], label=f"{rule.title()} consensus", alpha=0.7)
#     plt.scatter(ref_np[0], ref_np[1], label="Benign mean")
#     plt.legend()
#     plt.title(f"Malicious perturbation (gamma={gamma:.3f})")
#     plt.savefig("fig.jpg")
