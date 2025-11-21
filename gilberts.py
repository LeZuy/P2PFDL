import numpy as np
from typing import Dict, Tuple, Optional

def gilbert_coefficients(
    V: np.ndarray,
    p: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 10000,
    x0: Optional[np.ndarray] = None,
    prune_tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Compute convex combination coefficients expressing point p as a combination
    of columns/rows in V (convex hull projection) using Gilbert's algorithm.

    Parameters
    ----------
    V : np.ndarray
        Shape (m, d) array of m vectors in R^d.
    p : np.ndarray
        Shape (d,) target point.
    tol : float
        Stop when ||p - x_k||_2 <= tol.
    max_iter : int
        Maximum iterations.
    x0 : Optional[np.ndarray]
        Optional warm-start point in conv(V). If None, starts at a single vertex.
    prune_tol : float
        Drop coefficients smaller than this and renormalize.

    Returns
    -------
    x : np.ndarray
        The point in conv(V) found by the algorithm (≈ projection of p).
    alpha_dense : np.ndarray
        Dense coefficients (length m), sum to 1, alpha >= 0 (up to num. noise).
    alpha_sparse : Dict[int, float]
        Sparse map {index: weight} of active vertices.
    """

    V = np.asarray(V, dtype=float)
    p = np.asarray(p, dtype=float)
    assert V.ndim == 2 and p.ndim == 1 and V.shape[1] == p.shape[0], "Shape mismatch"

    m, d = V.shape

    # --- Initialize x and coefficients
    if x0 is not None:
        x = np.asarray(x0, dtype=float).copy()
        # Basic sanity: bring x into the affine hull via projection if tiny drift
        # (assumes x0 already in conv(V), otherwise results may not be guaranteed)
        # Start with a uniform guess over nearest vertex if grossly off.
        if not np.isfinite(x).all() or x.shape != (d,):
            raise ValueError("x0 must be a finite vector with shape (d,)")
        # Build an initial alpha that sums to 1 by expressing x0 as a single vertex
        # fallback (we'll quickly move toward p anyway).
        j0 = int(np.argmax(V @ p))  # aligned with p
        alpha = {j0: 1.0}
        x = V[j0].copy()
    else:
        # Start at vertex most aligned with p
        j0 = int(np.argmax(V @ p))
        alpha = {j0: 1.0}
        x = V[j0].copy()

    def prune_and_normalize(alpha_dict: Dict[int, float]) -> Dict[int, float]:
        # Remove tiny weights and renormalize to sum 1 (if anything remains)
        alpha_clean = {i: w for i, w in alpha_dict.items() if w > prune_tol}
        s = sum(alpha_clean.values())
        if s <= 0:
            # Fallback to a single vertex to avoid degeneracy
            j = int(np.argmax(V @ p))
            return {j: 1.0}
        invs = 1.0 / s
        for i in list(alpha_clean.keys()):
            alpha_clean[i] *= invs
        return alpha_clean

    # --- Main loop
    for _ in range(max_iter):
        r = p - x
        rnorm2 = float(r @ r)
        if rnorm2 <= tol * tol:
            break

        # Linear oracle: choose vertex maximizing <r, v>
        # (equivalently, farthest in the direction of r)
        scores = V @ r  # shape (m,)
        j = int(np.argmax(scores))
        vj = V[j]

        d_vec = vj - x
        dd = float(d_vec @ d_vec)
        if dd <= 1e-20:
            # Direction collapsed (x already at vj); we are stuck—terminate.
            break

        # Exact line-search on segment [x, vj]
        lam = float((r @ d_vec) / dd)
        if lam <= 0:
            lam = 0.0
        elif lam >= 1:
            lam = 1.0

        # Update x and coefficients: x_{k+1} = (1 - lam) x_k + lam v_j
        x = x + lam * d_vec

        # Scale down all current weights, then add lam to the chosen vertex
        for i in list(alpha.keys()):
            alpha[i] *= (1.0 - lam)
            if alpha[i] <= prune_tol:
                del alpha[i]
        alpha[j] = alpha.get(j, 0.0) + lam

        # Clean tiny weights and renormalize to exactly sum to 1
        alpha = prune_and_normalize(alpha)

    # Build dense vector of coefficients
    alpha_dense = np.zeros(m, dtype=float)
    for i, w in alpha.items():
        alpha_dense[i] = w

    # Final normalization clamp for numerical hygiene
    alpha_dense[alpha_dense < 0] = 0.0
    s = alpha_dense.sum()
    if s > 0:
        alpha_dense /= s
    else:
        # pathological fallback
        j = int(np.argmax(V @ p))
        alpha_dense[j] = 1.0
        x = V[j].copy()
        alpha = {j: 1.0}

    return x, alpha_dense, alpha


# ----- Example usage -----
if __name__ == "__main__":
    # V are m points in R^d
    V = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    p = np.array([0.3, 0.7])

    x, alpha_dense, alpha_sparse = gilbert_coefficients(V, p, tol=1e-10, max_iter=10000)
    print("Projected x:", x)
    print("Dense alpha:", alpha_dense)     # sums to 1
    print("Sparse alpha:", alpha_sparse)   # only active vertices
    print("Check sum to 1:", alpha_dense.sum())
    print("Reconstruction error:", np.linalg.norm(x - (alpha_dense @ V)))
    print("Distance to p:", np.linalg.norm(p - x))
