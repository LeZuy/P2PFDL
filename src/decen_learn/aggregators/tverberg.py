# src/decen_learn/aggregators/tverberg.py
from .base import BaseAggregator, AggregationResult
import numpy as np

class TverbergAggregator(BaseAggregator):
    """Tverberg centerpoint-based aggregator"""
    
    def __init__(self, num_byzantine: int = 0, max_reduce_iters: int = 20):
        super().__init__(num_byzantine)
        self.requires_projection = True
        self.max_reduce_iters = max_reduce_iters
    
    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        from ..tverberg.centerpoint import centerpoint_2d
        
        center, info = centerpoint_2d(
            vectors, 
            max_reduce_iters=self.max_reduce_iters
        )
        
        return AggregationResult(
            vector=center,
            metadata=info
        )
