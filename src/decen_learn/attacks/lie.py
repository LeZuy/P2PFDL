import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from aggregator import consensus

# Inverse CDF (quantile) for standard normal using Peter J. Acklam's approximation
# Source: https://web.archive.org/web/20150910064453/http://home.online.no/~pjacklam/notes/invnorm/

def required_supporters(n, m):
    return math.ceil(n/2 + 1) - m

def compute_z_max(n: int, m: int) -> float:
    s = required_supporters(n, m)
    benign = n
    if benign <= 0:
        raise ValueError("No benign workers (n - m <= 0).")
    target = (n - s) / benign
    # Clip numerical edge cases: ensure target in (0,1)
    target = min(max(target, 1e-12), 1 - 1e-12)
    z_max = norm.ppf(target)
    return s, target, z_max

def craft_malicious_vector(vectors: torch.Tensor, z:float, n:int, m:int):
    s, target, z = compute_z_max(n, m)
    print(f"s: {s}, target: {target}, z_max: {z}")
    mu = vectors.mean(axis=0)
    sigma = vectors.std(axis=0,unbiased=False)
    return mu + z * sigma

if __name__ == "__main__":
    torch.manual_seed(0)
    rule = "tverberg"
    vectors = torch.randn(6, 2)

    test_z = 1.0
    v_m = craft_malicious_vector(vectors=vectors, z=test_z, n=9, m=3)

    byzantine_vectors = torch.vstack([v_m] * 3)
    consensus_vec = consensus(torch.vstack([vectors, byzantine_vectors]), rule)

    benign_np = vectors.detach().cpu().numpy()
    byzantine_np = byzantine_vectors.detach().cpu().numpy()
    consensus_np = consensus_vec.detach().cpu().numpy()
    v_ref = vectors.mean(dim = 0)

    plt.figure(figsize=(6, 6))
    plt.scatter(benign_np[:, 0], benign_np[:, 1], label="Benign vectors", alpha=0.7)
    plt.scatter(byzantine_np[:, 0], byzantine_np[:, 1], label="Malicious vectors", alpha=0.7)
    plt.scatter(consensus_np[0], consensus_np[1], label=f"{rule.title()} consensus", alpha=0.7)
    plt.scatter(v_ref[0], v_ref[1], label=f"Mean", alpha=0.7)

    plt.legend()
    plt.title(f"LIE attack(z={test_z:.3f})")
    plt.savefig("fig.jpg")
