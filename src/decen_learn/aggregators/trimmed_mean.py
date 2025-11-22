# src/decen_learn/aggregators/trimmed_mean.py
import numpy as np
from .base import BaseAggregator, AggregationResult

class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed-mean aggregator"""
    def __init__(self, byzantine_fraction: float = 1/3):
        super().__init__(byzantine_fraction=byzantine_fraction)

    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        n, d = vectors.shape
        f = max(0, int(self.byzantine_fraction * n))
        
        if 2 * f >= n:
            f = max(0, (n - 1) // 2)
        
        result = []
        for j in range(d):
            sorted_vals = np.sort(vectors[:, j])
            trimmed = sorted_vals[f:n-f] if f > 0 else sorted_vals
            result.append(np.mean(trimmed))
        
        return AggregationResult(
            vector=np.array(result),
            metadata={"trimmed_count": f, "n_vectors": n}
        )
