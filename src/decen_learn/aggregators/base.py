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
    
    def __init__(
        self,
        num_byzantine: int = 0,
        byzantine_fraction: float = 0.33,
    ):
        # Allow passing either explicit count or fraction; algorithms can pick.
        self.num_byzantine = num_byzantine
        self.byzantine_fraction = byzantine_fraction
        # Most aggregators operate in the original weight space.
        # Set to True for aggregators that expect projected inputs (e.g., Tverberg).
        self.requires_projection: bool = False
    
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
        # If too few vectors, fallback to mean
        if vectors.shape[0] <= 2:
            return AggregationResult(
                vector=np.mean(vectors, axis=0),
                metadata={"fallback": "mean_two_vectors", "n_vectors": vectors.shape[0]},
            )
        return self.aggregate(vectors)
    
    def _validate_input(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {vectors.shape}")
        return vectors
