# src/decen_learn/aggregators/geomedian.py
import torch
from .base import BaseAggregator, AggregationResult
from geom_median.torch import compute_geometric_median

class GeoMedianAggregator(BaseAggregator):
    """Geo-Median aggregator"""
    
    def aggregate(self, vectors: torch.Tensor) -> AggregationResult:
        geomed_vec = compute_geometric_median(vectors)
        return AggregationResult(
            vector=geomed_vec,
            metadata={"n_vectors": int(vectors.shape[0])}
        )
