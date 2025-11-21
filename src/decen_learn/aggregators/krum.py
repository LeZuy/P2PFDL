# src/decen_learn/aggregators/krum.py
from aggregator import BaseAggregator, AggregationResult
import numpy as np

class KrumAggregator(BaseAggregator):
    """Multi-Krum aggregation rule."""
    
    def __init__(self, num_byzantine: int = 0, num_select: int = 1):
        super().__init__(num_byzantine)
        self.num_select = num_select
    
    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        m, d = vectors.shape
        f = self.num_byzantine
        
        if not (0 <= f < m // 2):
            raise ValueError(f"Invalid f={f} for m={m} vectors")
        
        # Compute pairwise distances efficiently
        scores = self._compute_scores(vectors, f)
        selected_idx = int(np.argmin(scores))
        
        return AggregationResult(
            vector=vectors[selected_idx].copy(),
            selected_index=selected_idx,
            metadata={"scores": scores}
        )
    
    def _compute_scores(self, vectors: np.ndarray, f: int) -> np.ndarray:
        m = vectors.shape[0]
        nb = m - f - 2
        
        # Efficient pairwise distance computation
        sq_norms = np.sum(vectors ** 2, axis=1, keepdims=True)
        dists = sq_norms + sq_norms.T - 2.0 * (vectors @ vectors.T)
        np.fill_diagonal(dists, 0.0)
        np.maximum(dists, 0.0, out=dists)
        
        if nb <= 0:
            return dists.sum(axis=1)
        
        # Sum of nb smallest distances for each vector
        scores = np.zeros(m)
        for i in range(m):
            d_i = np.delete(dists[i], i)
            scores[i] = np.sum(np.partition(d_i, nb - 1)[:nb])
        
        return scores