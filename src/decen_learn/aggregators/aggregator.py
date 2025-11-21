# src/decen_learn/aggregators/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class AggregationResult:
    """Standardized output from all aggregators."""
    vector: np.ndarray
    selected_index: Optional[int] = None  # For selection-based methods like Krum
    weights: Optional[np.ndarray] = None  # Convex combination weights
    metadata: dict = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}

class BaseAggregator(ABC):
    """Abstract base class for all aggregation methods."""
    
    def __init__(self, num_byzantine: int = 0):
        self.num_byzantine = num_byzantine
    
    @abstractmethod
    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        """
        Aggregate input vectors.
        
        Args:
            vectors: Shape (n, d) array of n vectors in d dimensions
            
        Returns:
            AggregationResult containing the aggregated vector and metadata
        """
        pass
    
    def __call__(self, vectors: np.ndarray) -> AggregationResult:
        vectors = self._validate_input(vectors)
        return self.aggregate(vectors)
    
    def _validate_input(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")
        return vectors


# src/decen_learn/aggregators/krum.py
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


# src/decen_learn/aggregators/tverberg.py
class TverbergAggregator(BaseAggregator):
    """Tverberg centerpoint-based aggregation."""
    
    def __init__(self, num_byzantine: int = 0, max_reduce_iters: int = 20):
        super().__init__(num_byzantine)
        self.max_reduce_iters = max_reduce_iters
    
    def aggregate(self, vectors: np.ndarray) -> AggregationResult:
        from ..projection.tverberg import centerpoint_2d
        
        center, info = centerpoint_2d(
            vectors, 
            max_reduce_iters=self.max_reduce_iters
        )
        
        return AggregationResult(
            vector=center,
            metadata=info
        )


# src/decen_learn/aggregators/__init__.py
from typing import Dict, Type
from .base import BaseAggregator, AggregationResult
from .krum import KrumAggregator
from .tverberg import TverbergAggregator
from .mean import MeanAggregator
from .trimmed_mean import TrimmedMeanAggregator

AGGREGATORS: Dict[str, Type[BaseAggregator]] = {
    "mean": MeanAggregator,
    "krum": KrumAggregator,
    "tverberg": TverbergAggregator,
    "trimmed_mean": TrimmedMeanAggregator,
}

def get_aggregator(name: str, **kwargs) -> BaseAggregator:
    """Factory function for creating aggregators."""
    if name not in AGGREGATORS:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(AGGREGATORS.keys())}")
    return AGGREGATORS[name](**kwargs)