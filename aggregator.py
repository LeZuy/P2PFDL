import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tverberg import centerpoint_2d
from geom_median.numpy import compute_geometric_median

def geo_median(vectors):
    return compute_geometric_median(vectors).median

def trimmed_mean(vectors, f):
    """
    vectors: np.ndarray of shape (n, 2)
    f: number of elements being trimmed
    """
    n, d = vectors.shape
    assert 2*f < n, "f too large"
    result = []
    for j in range(d):
        sorted_vals = np.sort(vectors[:, j])
        trimmed = sorted_vals[f:n-f]
        result.append(np.mean(trimmed))
    return np.array(result)

def pseudo_krum(vectors, *, return_index: bool = False):
    D = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    scores = D.sum(axis=1)
    idx = int(np.argmin(scores))
    if return_index:
        return idx, vectors[idx]
    return vectors[idx]

def krum(vectors: np.ndarray, f: int, *, return_index: bool = False):
    """
    Krum aggregation.
    
    Args:
        vectors: np.ndarray of shape (m, d) where m = number of clients, d = vector dim
        f: int, upper bound on number of Byzantine (compromised) clients (c in paper)
           Krum assumes f < m/2 - 1 (practically f should be << m)
    
    Returns:
        selected_index: int index of the selected vector (the "aggregate")
        selected_vector: np.ndarray of shape (d,) the chosen vector
        scores: np.ndarray shape (m,) the Krum score for each vector (lower is better)
    """
    m, d = vectors.shape
    if not (0 <= f < m):
        raise ValueError("f must satisfy 0 <= f < m")
    # number of nearest neighbors to consider for scoring:
    nb = m - f - 2
    # pairwise squared Euclidean distances using Gram trick to avoid (m,m,d) tensor
    vecs = vectors.astype(np.float64, copy=False)
    sq_norms = np.sum(vecs * vecs, axis=1, keepdims=True)  # (m,1)
    dists = sq_norms + sq_norms.T - 2.0 * (vecs @ vecs.T)
    np.fill_diagonal(dists, 0.0)
    # numerical errors can introduce small negative values; clamp them
    np.maximum(dists, 0.0, out=dists)
    
    scores = np.zeros(m, dtype=float)
    if nb <= 0:
        warnings.warn(
            f"m = {m}, f = {f} values lead to non-positive number of neighbors (m - f - 2 must be > 0). "
            "Falling back to pseudo-Krum scoring."
        )
        scores = dists.sum(axis=1)
    else:
        for i in range(m):
            # distances from vector i to all others (exclude self)
            d_i = np.delete(dists[i], i)
            # find nb smallest distances and sum them
            nearest = np.partition(d_i, nb-1)[:nb]  # unsorted nearest nb distances
            scores[i] = np.sum(nearest)
    
    # pick the index with smallest score
    selected_index = int(np.argmin(scores))
    selected_vector = vectors[selected_index].copy()
    if return_index:
        return selected_index, selected_vector
    return selected_vector

@torch.no_grad()
def consensus(vectors: torch.Tensor, consensus_type: str) -> torch.Tensor:
    """
    Aggregate vectors according to consensus rule.

    Args:
        vectors: Tensor (n, d).
        consensus_type: Aggregation rule name (mean, krum, tverberg, trimmed_mean).
    """
    ct = consensus_type.lower()
    device = vectors.device
    dtype = vectors.dtype

    if ct == "mean":
        return vectors.mean(dim=0)
    
    vectors_np = vectors.detach().cpu().numpy()
    f = max(0, vectors_np.shape[0] // 3)

    if ct == "krum":
        aggregated = krum(vectors_np, f)
    elif ct == "trimmed_mean":
        aggregated = trimmed_mean(vectors_np, f)
    elif ct == "tverberg":
        aggregated, _ = centerpoint_2d(vectors_np)
    else:
        # fallback to a safe mean if unknown type
        return vectors.mean(dim=0)

    return torch.as_tensor(aggregated, device=device, dtype=dtype)

if __name__ == "__main__":
    np.random.seed(0)
    m = 10   # total clients
    d = 2 
    f = 3 

    benign_count = m - f
    benign = np.random.normal(loc=0.0, scale=0.1, size=(benign_count, d))

    byz = np.random.normal(loc=0, scale=0.1, size=(f, d))

    vectors = np.vstack([benign, byz])
    perm = np.random.permutation(m)
    vectors = vectors[perm]

    krum_selected_vector = krum(vectors, f=f)
    trmean_selected_vector = trimmed_mean(vectors, f=f)
    tverberg_selected_vector, _ = centerpoint_2d(vectors)

    print("Total clients (m):", m)
    print("Byzantine count (f):", f)
    print(f"Selected vector: \n + Krum: {krum_selected_vector} \
          \n + Trimmed Mean: {trmean_selected_vector} \
          \n + Tverberg: {tverberg_selected_vector}")

    plt.scatter(vectors[:,0], vectors[:, 1], alpha = 0.7, c="b")
    plt.scatter(krum_selected_vector[0], krum_selected_vector[1], c="orange", alpha = 0.7, label="krum")
    plt.scatter(trmean_selected_vector[0], trmean_selected_vector[1], c="g",alpha = 0.7, label="trmean")
    plt.scatter(tverberg_selected_vector[0], tverberg_selected_vector[1], c="r", marker="*",alpha = 0.7, label="tverberg")
    plt.legend()
    plt.savefig("aggregator.jpg")
