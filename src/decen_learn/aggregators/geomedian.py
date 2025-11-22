# src/decen_learn/aggregators/mean.py
import numpy as np
from .base import BaseAggregator, AggregationResult
from geom_median.torch import compute_geometric_median   # PyTorch API
# from geom_median.numpy import compute_geometric_median  # NumPy API
class GeoMedianAggregator(BaseAggregator):
    """Geo-Median aggregator"""
    
    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        geomed_vec = compute_geometric_median(vectors)
        return AggregationResult(
            vector=geomed_vec,
            metadata={"n_vectors": len(vectors)}
        )