# src/decen_learn/aggregators/tverberg.py
from .base import BaseAggregator, AggregationResult
import numpy as np
import torch

class TverbergAggregator(BaseAggregator):
    """Tverberg centerpoint-based aggregator"""
    
    def __init__(self, num_byzantine: int = 0, max_reduce_iters: int = 20):
        super().__init__(num_byzantine)
        self.requires_projection = True
        self.max_reduce_iters = max_reduce_iters
    
    def aggregate(self, vectors) -> AggregationResult:
        from ..tverberg.centerpoint import centerpoint_2d
        
        tensor = self._validate_input(vectors)
        device = tensor.device
        dtype = tensor.dtype
        center_np, info = centerpoint_2d(
            tensor.detach().cpu().numpy(),
            max_reduce_iters=self.max_reduce_iters
        )
        center_tensor = torch.as_tensor(center_np, dtype=dtype, device=device)
        
        return AggregationResult(
            vector=center_tensor,
            metadata=info
        )
