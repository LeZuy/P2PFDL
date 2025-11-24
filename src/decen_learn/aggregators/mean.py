# src/decen_learn/aggregators/mean.py
import torch
from .base import BaseAggregator, AggregationResult

class MeanAggregator(BaseAggregator):
    """Simple mean aggregator."""
    
    def aggregate(self, vectors: torch.Tensor) -> AggregationResult:
        mean_vec = vectors.mean(dim=0)
        return AggregationResult(
            vector=mean_vec,
            metadata={"n_vectors": int(vectors.shape[0])}
        )
