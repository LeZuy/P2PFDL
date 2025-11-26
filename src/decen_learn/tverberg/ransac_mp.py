import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np

from .ransac import ransac_simplex


def _run_ransac_chunk(
    X: np.ndarray,
    q: Optional[np.ndarray],
    mode: str,
    iterations: int,
    eps: float,
    min_inliers: int,
    seed: int,
    early_stop_rounds: Optional[int],
    batch_size: int,
):
    """Worker helper executed in a separate process."""
    rng = np.random.default_rng(seed)
    return ransac_simplex(
        X,
        q=q,
        mode=mode,
        iterations=iterations,
        eps=eps,
        min_inliers=min_inliers,
        rng=rng,
        early_stop_rounds=early_stop_rounds,
        batch_size=batch_size,
    )


def ransac_simplex_mp(
    X,
    q=None,
    mode: str = "contain_q",
    iterations: int = 2000,
    eps: float = 0.0,
    min_inliers: int = 0,
    rng: Optional[np.random.Generator] = None,
    early_stop_rounds: Optional[int] = None,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
):
    """
    Multiprocessing wrapper around ransac_simplex. Splits iterations across workers.
    """
    X = np.asarray(X, dtype=float)
    q = None if q is None else np.asarray(q, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, int(num_workers))

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    tasks = []
    chunk = int(math.ceil(iterations / num_workers))
    seeds = rng.integers(low=0, high=2**63 - 1, size=num_workers, dtype=np.uint64)
    total_assigned = 0
    for worker_id in range(num_workers):
        remaining = iterations - total_assigned
        if remaining <= 0:
            break
        worker_iters = min(chunk, remaining)
        total_assigned += worker_iters
        tasks.append(
            (
                X,
                q,
                mode,
                worker_iters,
                eps,
                min_inliers,
                int(seeds[worker_id]),
                early_stop_rounds,
                batch_size,
            )
        )

    best_result = None
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(_run_ransac_chunk, *args) for args in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result

    return best_result


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    d = 3
    X = rng.normal(size=(n, d))
    w = rng.random(n)
    w /= w.sum()
    q = w @ X

    mp_result = ransac_simplex_mp(
        X,
        q=q,
        iterations=2000,
        eps=1e-6,
        batch_size=32,
        num_workers=4,
        rng=rng,
    )
    print("MP RANSAC success:", mp_result["success"])
    print("Best score:", mp_result["score"])
