# src/decen_learn/aggregators/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class AggregationResult:
    """Standardized output from all aggregators."""
    vector: torch.Tensor
    selected_index: Optional[int] = None  # For selection-based methods like Krum
    weights: Optional[torch.Tensor] = None  # Convex combination weights
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
    def aggregate(self, vectors: torch.Tensor) -> AggregationResult:
        """
        Aggregate input vectors.
        
        Args:
            vectors: Shape (n, d) array of n vectors in d dimensions
            
        Returns:
            AggregationResult containing the aggregated vector and metadata
        """
        pass
    
    def __call__(self, vectors) -> AggregationResult:
        tensor = self._validate_input(vectors)
        # If too few vectors, fallback to mean
        if tensor.shape[0] <= 2:
            return AggregationResult(
                vector=tensor.mean(dim=0),
                metadata={
                    "fallback": "mean_two_vectors",
                    "n_vectors": int(tensor.shape[0]),
                },
            )
        return self.aggregate(tensor)
    
    def _validate_input(self, vectors) -> torch.Tensor:
        if torch.is_tensor(vectors):
            tensor = vectors
        else:
            tensor = torch.as_tensor(vectors)
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {tuple(tensor.shape)}")
        return tensor
