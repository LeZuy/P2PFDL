import numpy as np

from decen_learn.tverberg.centerpoint import centerpoint_2d, monotone_chain_hull
from decen_learn.tverberg.ransac import ransac_simplex


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a 2D point lies inside or on the boundary of a convex polygon."""
    if len(polygon) == 0:
        return False
    # Use cross-product sign consistency for convex polygons
    signs = []
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        da = b - a
        dp = point - a
        cross_z = da[0] * dp[1] - da[1] * dp[0]
        signs.append(np.sign(cross_z))
    signs = np.array(signs)
    # Allow points on edges (zeros) and require non-conflicting signs
    return np.all(signs >= -1e-12) or np.all(signs <= 1e-12)


def test_centerpoint_inside_convex_hull():
    # Arrange: square with an interior cluster
    base = np.array([
        [2.5, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 3.6],
    ])
    rng = np.random.default_rng(0)
    interior = rng.uniform(0.25, 0.75, size=(20, 2))
    points = np.vstack([base, interior])

    center, _ = centerpoint_2d(points)
    hull = monotone_chain_hull(points)

    assert _point_in_polygon(center, hull), "Centerpoint must lie inside the convex hull"


def test_ransac_recovers_convex_weights_for_query_point():
    # Simple 2D simplex (triangle)
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    q = np.array([0.2, 0.2])

    result = ransac_simplex(X, q=q, mode="contain_q", iterations=10, rng=np.random.default_rng(1))

    assert result["success"], "RANSAC should find a feasible simplex"
    # Dense weights should reconstruct the query point when they exist
    weights = result["q_weights_dense"]
    assert weights is not None
    assert np.isclose(weights.sum(), 1.0)
    reconstructed = weights @ X
    np.testing.assert_allclose(reconstructed, q, atol=1e-6)