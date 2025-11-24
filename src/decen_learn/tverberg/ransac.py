import warnings

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linprog

def _affine_independent(verts, tol=1e-12):
    """
    verts: (d+1, d). Return True if vertices are affinely independent.
    Uses QR instead of SVD-based rank to trim overhead.
    """
    d = verts.shape[1]
    if verts.shape[0] != d + 1:
        return False
    A = (verts[:-1] - verts[-1]).T  # shape (d, d)
    try:
        _, r = np.linalg.qr(A)
    except np.linalg.LinAlgError:
        return False
    return np.all(np.abs(np.diag(r)) > tol)

def _compute_barycentric_matrix(points, verts, tol=1e-12):
    """
    Compute barycentric coordinates for multiple points with respect to verts.
    Returns array (n_points, d+1) or None if simplex is singular.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    d = verts.shape[1]
    if pts.shape[1] != d:
        raise ValueError("Points must have same dimensionality as verts.")
    A = (verts[:-1] - verts[-1]).T  # (d, d)
    rhs = pts - verts[-1]  # (n, d)
    try:
        w = np.linalg.solve(A, rhs.T).T  # (n, d)
    except np.linalg.LinAlgError:
        return None
    lam = np.empty((pts.shape[0], d + 1), dtype=float)
    lam[:, :-1] = w
    lam[:, -1] = 1.0 - np.sum(w, axis=1)
    lam[np.abs(lam) < tol] = 0.0
    return lam


def _barycentric_coords(x, verts, tol=1e-12):
    """
    Compute barycentric coordinates of point x wrt simplex defined by verts.
    Returns lambda (d+1,), or None if simplex is degenerate.
    """
    lam = _compute_barycentric_matrix(x, verts, tol=tol)
    if lam is None:
        return None
    return lam[0]

def qp_optimization(verts, q, tol=1e-12):
    try:
        n , d= verts.shape
        lam = cp.Variable(n)

        constraints = [
            verts.T @ lam == q,
            cp.sum(lam) == 1,
            lam >= 0              # or lam >= 1e-6 / n for strict positivity
        ]

        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(lam)), constraints)
        prob.solve(verbose=False)

        print(f"Lambda_star: {lam.value}") # barycentric weights
        return lam.value

    except Exception as e:
        return None
    

def _in_simplex(x, verts, eps=0.0, tol=1e-12):
    """
    Check if x is inside simplex conv(verts) with slack eps on barycentric >= -eps.
    """
    lam = _barycentric_coords(x, verts, tol=tol)
    if lam is None:
        return False
    return (lam >= -eps - tol).all() and (abs(lam.sum() - 1.0) <= 10 * tol)


def _in_simplex_batch(X, verts, eps=0.0, tol=1e-12, return_lam=False):
    """
    Vectorized check if each point in X is inside simplex conv(verts).
    X: (n, d) or (d,)
    verts: (d+1, d)
    Returns boolean array (n,) and optionally barycentric coordinates.
    """
    pts = np.atleast_2d(np.asarray(X, dtype=float))
    lam = _compute_barycentric_matrix(pts, verts, tol=tol)
    if lam is None:
        mask = np.zeros(pts.shape[0], dtype=bool)
        return (mask, None) if return_lam else mask
    tol_sum = 10.0 * tol
    mask = (lam >= -eps - tol).all(axis=1) & (np.abs(np.sum(lam, axis=1) - 1.0) <= tol_sum)
    if return_lam:
        return mask, lam
    return mask

def dense_simplex_weights(X, verts_idx, q, tol=1e-12):
    """Return a dense coefficient vector w of length n such that
    q = sum_i w[i] * X[i] if q lies in the simplex conv(X[verts_idx]).
    Coefficients are zero outside verts_idx. Returns None if simplex is
    degenerate or q is not representable within numerical tolerance.

    Parameters
    ----------
    X : (n,d) array
    verts_idx : (d+1,) int indices into X selecting simplex vertices
    q : (d,) point
    tol : float numerical tolerance for barycentric solution
    """
    X = np.asarray(X, dtype=float)
    q = np.asarray(q, dtype=float)
    verts = X[np.asarray(verts_idx, dtype=int)]
    lam = _barycentric_coords(q, verts, tol=tol)
    if lam is None:
        return None
    # Build dense coefficient vector
    w = np.zeros(len(X), dtype=float)
    w[np.asarray(verts_idx, dtype=int)] = lam
    return w

def inverse_distance_weights(q, verts, p: float = 1.0, eps: float = 1e-12):
    """Compute inverse-distance weights of q to each vertex in verts.
    Returns a vector w (len = d+1) with w_i ∝ 1 / ||q - v_i||^p and sum(w)=1.
    If q coincides with any vertex (distance < eps), assigns equal weight among
    coincident vertices and zero to others.
    """
    q = np.asarray(q, dtype=float)
    verts = np.asarray(verts, dtype=float)
    dists = np.linalg.norm(verts - q[None, :], axis=1)
    near = dists < eps
    if np.any(near):
        k = int(np.sum(near))
        w = np.zeros_like(dists)
        w[near] = 1.0 / k
        return w
    # Standard IDW
    w = 1.0 / (dists ** p)
    w_sum = np.sum(w)
    if w_sum <= 0:
        # fallback uniform
        return np.full_like(w, 1.0 / len(w))
    return w / w_sum

def dense_idw_weights(X, verts_idx, q, p: float = 1.0, eps: float = 1e-12):
    """Dense coefficient vector based on inverse-distance weights over verts_idx.
    Returns w (len n) with zeros outside verts_idx, normalized to sum to 1.
    """
    X = np.asarray(X, dtype=float)
    verts_idx = np.asarray(verts_idx, dtype=int)
    verts = X[verts_idx]
    wv = inverse_distance_weights(q, verts, p=p, eps=eps)
    w = np.zeros(len(X), dtype=float)
    w[verts_idx] = wv
    return w

def ransac_simplex(
    X,
    q=None,
    mode="contain_q",      # "contain_q" or "cover_X"
    iterations=2000,
    eps=0.0,
    min_inliers=0,
    rng=None,
    early_stop_rounds=None,
    batch_size=16
):
    """
    RANSAC to find a d+1-vertex simplex that maximizes inliers.
    - X: (n, d) data
    - q: (d,) optional query point (required if mode="contain_q")
    - mode:
        * "contain_q": simplex must contain q; score = #X contained
        * "cover_X":   no q; score = #X contained
    - iterations: number of random trials
    - batch_size: number of random simplices evaluated together
    - eps: inlier slack on barycentric coords (>= -eps)
    - min_inliers: optional minimal score to accept early
    - rng: np.random.Generator for reproducibility
    - early_stop_rounds: stop if no improvement after this many trials

        Returns dict with:
            'verts': (d+1, d) best vertices,
            'verts_idx': (d+1,) indices of these vertices in X,
            'inliers_idx': np.array of indices of inliers,
            'score': best score,
            'success': bool,
            'q_barycentric': (d+1,) barycentric coords of q in 'verts' (if q provided),
            'q_weights_dense': (n,) dense barycentric weights (zeros except at verts_idx),
            'q_idw_verts': (d+1,) inverse-distance weights on verts (sum=1),
            'q_idw_point': (d,) point from IDW average of verts,
            'q_idw_weights_dense': (n,) dense IDW weights aligned to X
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if mode not in ("contain_q", "cover_X"):
        raise ValueError("mode must be 'contain_q' or 'cover_X'")
    if mode == "contain_q" and q is None:
        raise ValueError("q must be provided in 'contain_q' mode")
    if q is not None:
        q = np.asarray(q, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    best = {'score': -1, 'verts': None, 'verts_idx': None, 'inliers_idx': None,
        'q_barycentric': None, 'q_weights_dense': None,
        'q_idw_verts': None, 'q_idw_point': None, 'q_idw_weights_dense': None}
    no_improve = 0

    batch_size = max(1, int(batch_size))
    trials = 0
    tol = 1e-12
    while trials < iterations:
        current_batch = min(batch_size, iterations - trials)
        trials += current_batch

        idx_batch = np.empty((current_batch, d + 1), dtype=int)
        for b in range(current_batch):
            idx_batch[b] = rng.choice(n, size=d+1, replace=False)
        verts_batch = X[idx_batch]  # (batch, d+1, d)

        # Filter affine-independent simplices
        affine_mask = np.array([_affine_independent(v) for v in verts_batch], dtype=bool)
        if not affine_mask.any():
            continue
        verts_batch = verts_batch[affine_mask]
        idx_batch = idx_batch[affine_mask]

        # Construct A matrices (batch, d, d)
        A = np.transpose(verts_batch[:, :-1, :] - verts_batch[:, -1:, :], (0, 2, 1))

        # Ensure q lies in simplex when required
        if mode == "contain_q":
            rhs_q = (q - verts_batch[:, -1, :])  # (batch, d)
            try:
                w_q = np.linalg.solve(A, rhs_q[..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                warnings.warn("Failed to solve barycentric system for q; skipping batch.")
                continue
            lam_q = np.empty((w_q.shape[0], d + 1), dtype=float)
            lam_q[:, :-1] = w_q
            lam_q[:, -1] = 1.0 - np.sum(w_q, axis=1)
            contain_mask = (lam_q >= -eps - tol).all(axis=1) & (np.abs(np.sum(lam_q, axis=1) - 1.0) <= 10.0 * tol)
            if not contain_mask.any():
                continue
            verts_batch = verts_batch[contain_mask]
            idx_batch = idx_batch[contain_mask]
            A = A[contain_mask]

        if len(verts_batch) == 0:
            continue

        # Solve for barycentric coords of all X for each simplex in batch
        rhs = X[None, :, :] - verts_batch[:, None, -1, :]  # (batch, n, d)
        rhs = np.transpose(rhs, (0, 2, 1))  # (batch, d, n)
        try:
            w_all = np.linalg.solve(A, rhs)  # (batch, d, n)
        except np.linalg.LinAlgError:
            warnings.warn("Failed to solve barycentric system for samples; skipping batch.")
            continue
        w_all = np.transpose(w_all, (0, 2, 1))  # (batch, n, d)
        lam = np.empty((w_all.shape[0], n, d + 1), dtype=float)
        lam[..., :-1] = w_all
        lam[..., -1] = 1.0 - np.sum(w_all, axis=2)
        inlier_mask = (lam >= -eps - tol).all(axis=2) & (np.abs(np.sum(lam, axis=2) - 1.0) <= 10.0 * tol)
        scores = inlier_mask.sum(axis=1)

        if scores.size == 0:
            continue
        best_idx_in_batch = int(np.argmax(scores))
        score = int(scores[best_idx_in_batch])
        inliers = np.flatnonzero(inlier_mask[best_idx_in_batch])

        if score > best['score']:
            best['score'] = score
            best['verts'] = verts_batch[best_idx_in_batch].copy()
            best['verts_idx'] = idx_batch[best_idx_in_batch].copy()
            best['inliers_idx'] = inliers
            no_improve = 0
        else:
            no_improve += current_batch

        # Early exit checks
        if min_inliers and best['score'] >= min_inliers:
            break
        if early_stop_rounds and no_improve >= early_stop_rounds:
            break

    best['success'] = best['score'] >= 0
    if q is not None and best['inliers_idx'] is not None and best['inliers_idx'].size:
        inlier_points = X[best['inliers_idx']]
        mask = ~np.isclose(
            inlier_points[:, None, :],
            best['verts'][None, :, :]
        ).all(axis=2).any(axis=1)
        if not np.any(mask):
            mask = np.ones(len(inlier_points), dtype=bool)
        inlier_points_excl_verts = inlier_points[mask]
        best['inliers_excl_verts'] = inlier_points_excl_verts

        best['inliers_excl_verts_indices'] = best['inliers_idx'][mask]
        lam = qp_optimization(inlier_points_excl_verts, q)
        best['q_barycentric'] = lam
        if lam is not None:
            idx = best['inliers_excl_verts_indices']
            w_dense = np.zeros(n)
            w_dense[idx] = lam
            best['q_weights_dense'] = w_dense
        else:
            w_dense = np.zeros(n)
            w_dense[best['inliers_excl_verts_indices']] = 1 / len(best['inliers_excl_verts_indices'])
            best['q_weights_dense'] = w_dense
            
    return best

if __name__ == "__main__":
    d = 1000
    rng = np.random.default_rng()
    n = 15
    Xd = rng.uniform(size=(n, d))  
    P = rng.normal(size=(2, d))  
    X2 = Xd @ P.T 

    w_true = rng.random(n)  # a random convex combination
    w_true /= w_true.sum()
    qd_true = w_true @ Xd
    q2_proj = w_true @ X2
    
    plt.scatter(X2[:,0], X2[:,1], c="b", alpha=0.5)
    plt.scatter(q2_proj[0], q2_proj[1], c="r", alpha = 0.5)

    print("True barycentric weights:", np.round(w_true, 3))
    print("True 3D point:", np.round(qd_true, 4))
    print("Projected 2D point:", np.round(q2_proj, 4))

    result = ransac_simplex(
        X2, q=q2_proj, mode="contain_q",
        iterations=5000, eps=1e-9, rng=rng, early_stop_rounds=1000
    )

    print("\n=== RANSAC result ===")
    print("Success:", result["success"])
    print("Inliers:", result["inliers_idx"])
    print("Vertices (2D):\n", result["verts"])
    plt.scatter(result['verts'][:,0], result['verts'][:,1], c="black", alpha=0.5)
    plt.scatter(X2[result['inliers_idx'],0], X2[result['inliers_idx'],1], c="green", alpha=0.5 )
    plt.savefig("ransac.jpg")

    print("Barycentric on simplex:", np.round(result["q_barycentric"], 4))

    w_ransac = result["q_weights_dense"]
    print("Dense convex weights:", np.round(w_ransac, 3))

    qd_recon = w_ransac @ Xd

    print("\nReconstructed 3D point:", np.round(qd_recon, 4))
    print("Reconstruction error ||q3_true - q3_recon|| =", np.linalg.norm(qd_true - qd_recon))
    print(f"Consensus point is in convex hull:{is_in_convex_hull(qd_recon, Xd)}")