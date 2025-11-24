# src/decen_learn/aggregators/trimmed_mean.py
import torch
from .base import BaseAggregator, AggregationResult

class TrimmedMeanAggregator(BaseAggregator):
    """Trimmed-mean aggregator"""
    def __init__(self, byzantine_fraction: float = 1/3):
        super().__init__(byzantine_fraction=byzantine_fraction)

    def aggregate(self, vectors: torch.Tensor) -> AggregationResult:
        n, d = vectors.shape
        f = max(0, int(self.byzantine_fraction * n))
        
        if 2 * f >= n:
            f = max(0, (n - 1) // 2)
        
        if f == 0:
            trimmed_mean = vectors.mean(dim=0)
        else:
            sorted_vals, _ = torch.sort(vectors, dim=0)
            trimmed = sorted_vals[f:n-f]
            trimmed_mean = trimmed.mean(dim=0)
        
        return AggregationResult(
            vector=trimmed_mean,
            metadata={"trimmed_count": int(f), "n_vectors": int(n)}
        )
