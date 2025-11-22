# src/decen_learn/aggregators/mean.py
import numpy as np
from .base import BaseAggregator, AggregationResult

class MeanAggregator(BaseAggregator):
    """Simple mean aggregator."""
    
    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        mean_vec = np.mean(vectors, axis=0)
        return AggregationResult(
            vector=mean_vec,
            metadata={"n_vectors": len(vectors)}
        )