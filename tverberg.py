from __future__ import annotations
import math
from typing import Iterable, Tuple, List, Optional
import numpy as np

# ----------------------------
# Geometry utilities
# ----------------------------

def monotone_chain_hull(P: np.ndarray) -> np.ndarray:
    """Andrew monotone chain convex hull. Returns hull vertices CCW (last!=first)."""
    P = np.asarray(P, dtype=float)
    if len(P) <= 1:
        return P.copy()
    # sort lexicographically by x then y
    S = P[np.lexsort((P[:,1], P[:,0]))]
    def cross(o, a, b):  # 2D cross product (OA x OB)
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in S:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in S[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def clip_polygon_halfspace(poly: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """
    Clip convex (or general) polygon with halfspace {x : a·x <= b}.
    Returns possibly empty polygon.
    """
    if poly.size == 0:
        return poly
    res = []
    m = len(poly)
    def inside(v): return float(np.dot(a, v)) <= b + 1e-12
    for i in range(m):
        curr, nxt = poly[i], poly[(i+1)%m]
        inc, inn = inside(curr), inside(nxt)
        if inc and inn:
            res.append(nxt)
        elif inc and not inn:
            # leaving: add intersection
            inter = segment_halfspace_intersection(curr, nxt, a, b)
            if inter is not None:
                res.append(inter)
        elif (not inc) and inn:
            # entering: add intersection + next
            inter = segment_halfspace_intersection(curr, nxt, a, b)
            if inter is not None:
                res.append(inter)
            res.append(nxt)
        # else both out -> add nothing
    return np.array(res, dtype=float)

def segment_halfspace_intersection(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: float) -> Optional[np.ndarray]:
    """Intersection of segment pq with line a·x = b, assuming ends are on opposite sides."""
    d = q - p
    denom = float(np.dot(a, d))
    if abs(denom) < 1e-18:
        return None
    t = (b - float(np.dot(a, p))) / denom
    return p + t * d

def polygon_area_centroid(poly: np.ndarray) -> Tuple[float, np.ndarray]:
    """Signed area (positive if CCW) and centroid of polygon."""
    if len(poly) < 3:
        if len(poly) == 0:
            return 0.0, np.array([np.nan, np.nan])
        if len(poly) == 1:
            return 0.0, poly[0]
        # two points: return midpoint
        return 0.0, 0.5*(poly[0] + poly[1])
    x = poly[:,0]; y = poly[:,1]
    s = x*np.roll(y,-1) - y*np.roll(x,-1)
    A = 0.5*np.sum(s)
    if abs(A) < 1e-18:
        # degenerate; fall back to average
        return 0.0, np.mean(poly, axis=0)
    cx = np.sum((x + np.roll(x,-1))*s) / (6*A)
    cy = np.sum((y + np.roll(y,-1))*s) / (6*A)
    return A, np.array([cx, cy])

# ----------------------------
# Reduction step per the sketch
# ----------------------------

def quadrant_sets(S: np.ndarray, mx: float, my: float):
    """Return indices for LU, LD, RU, RD (inclusive of lines)."""
    x, y = S[:,0], S[:,1]
    L = x <= mx; R = x >= mx
    U = y >= my; D = y <= my
    LU = np.where(L & U)[0]
    LD = np.where(L & D)[0]
    RU = np.where(R & U)[0]
    RD = np.where(R & D)[0]
    return LU, LD, RU, RD

def replace_points_by_quadrant_hulls(S: np.ndarray, LU, LD, RU, RD) -> np.ndarray:
    """Keep only hull vertices in each non-empty quadrant."""
    # print(f"  Quadrant sizes: LU={len(LU)}, LD={len(LD)}, RU={len(RU)}, RD={len(RD)}")
    keep = []
    for idxs in (LU, LD, RU, RD):
        if len(idxs) == 0:
            continue
        H = monotone_chain_hull(S[idxs])
        if len(H) == 0:
            continue
        # map hull vertices back to original points (by value); robust via KD nearest
        # since hull points are subset of quadrant points, we can match with tolerance
        Q = S[idxs]
        for v in H:
            # find one match (first within tiny tolerance)
            diffs = np.linalg.norm(Q - v, axis=1)
            j = np.argmin(diffs)
            keep.append(idxs[j])
    keep = sorted(set(keep))
    # print(f"  After hull replacement: {len(keep)} points remain: {keep}")
    return S[keep]

# ----------------------------
# Tukey region (depth ≥ k) via strip intersection
# ----------------------------

def tukey_region_polygon(S: np.ndarray, k: int, m_angles: int = 181) -> np.ndarray:
    """
    Compute (approximate) convex polygon of points with halfspace depth ≥ k
    by intersecting strips along m_angles directions in [0, pi).
    Returns empty array if infeasible at this discretization.
    """
    S = np.asarray(S, dtype=float)
    if len(S) == 0:
        return S
    # start from a huge bounding box around points
    xmin, ymin = np.min(S, axis=0) - 10.0
    xmax, ymax = np.max(S, axis=0) + 10.0
    poly = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]], dtype=float)

    n = len(S)
    # Ensure k is feasible for current dataset size
    k_eff = min(k, n)
    if k_eff <= 0:
        return np.array([])
    
    for th in np.linspace(0.0, math.pi, m_angles, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)], dtype=float)  # unit direction
        proj = S @ u
        # lower bound: k-th smallest; upper bound: k-th largest (== nth - (k-1)-th smallest)
        lb = np.partition(proj, k_eff-1)[k_eff-1]
        if n - k_eff >= 0:
            ub = np.partition(proj, n-k_eff)[n-k_eff]
        else:
            ub = lb  # fallback when k is too large
        # intersect with u·x <= ub  and  -u·x <= -lb  (i.e., u·x >= lb)
        poly = clip_polygon_halfspace(poly, u, ub)
        if len(poly) == 0:
            return poly
        poly = clip_polygon_halfspace(poly, -u, -lb)
        if len(poly) == 0:
            return poly
    return poly

def tukey_region_polygon_pairs(S: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the exact Tukey depth-≥k region in 2D using all directions
    orthogonal to segments between pairs of input points (plus a few axes).
    This is exact for 2D because the arrangement of supporting lines changes
    only when the direction crosses a pairwise normal.
    """
    S = np.asarray(S, dtype=float)
    n = len(S)
    if n == 0:
        return np.empty((0,2))
    # Large initial box
    xmin, ymin = np.min(S, axis=0) - 10.0
    xmax, ymax = np.max(S, axis=0) + 10.0
    poly = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]], dtype=float)

    # Gather unique directions (normals to pair segments), up to sign
    seen = set()
    dirs = []
    for i in range(n):
        for j in range(i+1, n):
            v = S[j] - S[i]
            nv = float(np.linalg.norm(v))
            if nv < 1e-15:
                continue
            u = np.array([-v[1]/nv, v[0]/nv])  # perp unit vector
            # canonicalize sign to deduplicate
            if abs(u[0]) > 1e-12:
                if u[0] < 0: u = -u
            else:
                if u[1] < 0: u = -u
            key = tuple(np.round(u, 12))
            if key in seen:
                continue
            seen.add(key)
            dirs.append(u)

    # add a few fixed axes for robustness
    for u in (np.array([1.0,0.0]), np.array([0.0,1.0]),
              np.array([1.0,1.0])/np.sqrt(2), np.array([1.0,-1.0])/np.sqrt(2)):
        key = tuple(np.round(u, 12))
        if key not in seen:
            seen.add(key); dirs.append(u)

    nS = len(S)
    k_eff = min(k, nS)
    if k_eff <= 0:
        return np.empty((0,2))

    for u in dirs:
        proj = S @ u
        lb = np.partition(proj, k_eff-1)[k_eff-1]
        ub = np.partition(proj, nS-k_eff)[nS-k_eff] if nS-k_eff >= 0 else lb
        poly = clip_polygon_halfspace(poly, u, ub)
        if len(poly) == 0:
            return poly
        poly = clip_polygon_halfspace(poly, -u, -lb)
        if len(poly) == 0:
            return poly
    return poly

def estimate_tukey_depth(x: np.ndarray, S: np.ndarray, m_angles: int = 721) -> int:
    """Estimate Tukey depth of x by sampling many directions."""
    S = np.asarray(S, dtype=float)
    counts = []
    for th in np.linspace(0.0, math.pi, m_angles, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)], dtype=float)
        proj = S @ u
        t = float(x @ u)
        # how many points can lie in a closed halfspace with boundary orthogonal to u and containing x?
        # that is min(# <= t, # >= t)
        le = int(np.sum(proj <= t + 1e-12))
        ge = int(np.sum(proj >= t - 1e-12))
        counts.append(min(le, ge))
    return int(min(counts))

# ----------------------------
# Main: CENTERPOINT(S)
# ----------------------------

def centerpoint_2d(S: np.ndarray,
                   max_reduce_iters: int = 20,
                   hull_when_leq: int = 4,
                   angles_region: int = 181) -> Tuple[np.ndarray, dict]:
    """
    Compute a 2D centerpoint candidate for point set S (n x 2).
    Returns (point, info_dict).
    The algorithm follows the provided sketch:
      - repeat: compute median half-planes L,U,D,R; keep hull vertices per quadrant
      - then compute a depth-≥ceil(n/3) region via strip intersection and return its centroid
    info_dict contains diagnostic details.
    """
    S = np.asarray(S, dtype=float)
    n0 = len(S)
    Sred = S.copy()

    # Reduction
    history_sizes = [len(Sred)]
    for it in range(max_reduce_iters):
        if len(Sred) <= hull_when_leq:
            break
        mx = float(np.median(Sred[:,0]))
        my = float(np.median(Sred[:,1]))
        LU, LD, RU, RD = quadrant_sets(Sred, mx, my)
        Snext = replace_points_by_quadrant_hulls(Sred, LU, LD, RU, RD)
        if len(Snext) >= len(Sred):  # no progress -> stop
            break
        Sred = Snext
        history_sizes.append(len(Sred))

    # print(f"After reduction: {len(Sred)} points remain (started with {n0})")
    # Final region of depth ≥ ceil(n/3)
    k = int(math.ceil(n0/3))
    # For small or nearly collinear sets, prefer exact pairwise directions on ORIGINAL S
    hull = monotone_chain_hull(S)
    _, hull_centroid = polygon_area_centroid(hull) if len(hull) >= 3 else (0.0, np.mean(S, axis=0))
    if len(S) <= 64 or (len(hull) < 3):
        poly = tukey_region_polygon_pairs(S, k)
    else:
        poly = tukey_region_polygon(S, k, m_angles=angles_region)
    # If region is empty at this discretization, relax angles or k slightly
    if len(poly) == 0:
        # Try denser angles
        poly = tukey_region_polygon(S, k, m_angles=361)
    if len(poly) == 0 and k > 1:
        poly = tukey_region_polygon(S, k-1, m_angles=361)
        k = k-1

    if len(poly) == 0:
        # Fallback: intersection point of median lines
        c = np.array([np.median(S[:,0]), np.median(S[:,1])], dtype=float)
        region_area, center = 0.0, c
    else:
        _, center = polygon_area_centroid(poly)
        region_area = abs(polygon_area_centroid(poly)[0])
    # print (f"Centerpoint candidate at {center}, region area {region_area:.4f}, target depth {k}")
    depth_est = estimate_tukey_depth(center, S, m_angles=1081)
    # If depth is suspiciously low (degenerate strip), choose a robust 1D center on the data:
    if depth_est < k:
        # principal direction
        X = S - S.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        v = vt[0]
        proj = S @ v
        order = np.argsort(proj)
        # Candidate 1: midpoint between k-th smallest and (n-k+1)-th smallest along v
        iL = k-1
        iR = len(S)-k
        iL = max(0, min(iL, len(S)-1))
        iR = max(0, min(iR, len(S)-1))
        a = S[order[iL]]; b = S[order[iR]]
        center1 = 0.5*(a+b)
        d1 = estimate_tukey_depth(center1, S, m_angles=2047)
        # Candidate 2: the median sample point along v
        iM = len(S)//2
        center2 = S[order[iM]]
        d2 = estimate_tukey_depth(center2, S, m_angles=2047)
        # Choose the better
        if d1 >= d2:
            center, depth_est = center1, d1
        else:
            center, depth_est = center2, d2

    info = dict(
        n_original=n0,
        n_after_reduction=len(Sred),
        reduction_sizes=history_sizes,
        k_target=k,
        region_area=region_area,
        depth_estimate=depth_est
    )
    return center, info

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # mix of clusters to stress-test
    A = rng.normal([0,0], 1.0, size=(200,2))
    B = rng.normal([6,1], 0.1, size=(50,2))
    # C = rng.normal([-2,5], 0.8, size=(40,2))
    S = np.array([[-0.125983, 0.769243],
                  [0.351250, 0.596170],
                  [1.006601, 0.547590],
                  [0.583886, 0.217105],
                  [0.234167, 0.397447]])
    # S = np.vstack([A, B])

    c, info = centerpoint_2d(S)
    # print("Centerpoint candidate:", c)
    # print("Diagnostics:", info)

    # # Plotting (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,6))
        plt.scatter(S[:,0], S[:,1], color='blue', s=10, alpha=0.5)
        # plt.scatter(B[:,0], B[:,1], color='yellow', s=10, alpha=0.5, label = f"{len(B)} bad points")
        plt.scatter(c[0], c[1], color='red', s=100, label='Centerpoint candidate')
        plt.title('2D Centerpoint Candidate via Exact Algorithm')
        plt.axis('equal')
        plt.legend()
        plt.show()
    except ImportError:
        pass