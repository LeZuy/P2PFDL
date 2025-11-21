"""Weight projection strategies for dimensionality reduction."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.random_projection import GaussianRandomProjection


class WeightProjector(ABC):
    """Protocol for weight projection strategies."""
    
    @abstractmethod
    def project(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Project high-dimensional weights to low-dimensional space.
        
        Args:
            weights: Dictionary mapping layer names to weight arrays
            
        Returns:
            Low-dimensional projection vector
        """
        pass
    
    @abstractmethod
    def get_projection_dim(self) -> int:
        """Return the dimensionality of the projected space."""
        pass
    
    def flatten_weights(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten weight dictionary to 1D vector."""
        return np.concatenate([w.flatten() for w in weights.values()])


class RandomWeightProjector(WeightProjector):
    """Random projection using Gaussian random matrices.
    
    Uses Johnson-Lindenstrauss lemma to preserve pairwise distances
    with high probability while reducing dimensionality.
    """
    
    def __init__(
        self,
        original_dim: int,
        projection_dim: int = 2,
        random_state: Optional[int] = None,
    ):
        """Initialize random projector.
        
        Args:
            original_dim: Original weight space dimension
            projection_dim: Target projection dimension (typically 2)
            random_state: Random seed for reproducibility
        """
        self.original_dim = original_dim
        self.projection_dim = projection_dim
        self.random_state = random_state
        
        # Build projection matrix
        self._projector = GaussianRandomProjection(
            n_components=projection_dim,
            random_state=random_state
        )
        
        # Fit with dummy data to initialize the projection matrix
        dummy = np.random.randn(1, original_dim)
        self._projector.fit(dummy)
        
        # Store the projection matrix
        self.projection_matrix = self._projector.components_
    
    def project(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Project weights to low-dimensional space.
        
        Args:
            weights: Weight dictionary
            
        Returns:
            Projected vector of shape (projection_dim,)
        """
        flat = self.flatten_weights(weights)
        
        if len(flat) != self.original_dim:
            raise ValueError(
                f"Weight dimension mismatch: expected {self.original_dim}, "
                f"got {len(flat)}"
            )
        
        # Project: result = P @ x where P is (proj_dim, orig_dim)
        projected = self.projection_matrix @ flat
        return projected
    
    def get_projection_dim(self) -> int:
        """Return projection dimensionality."""
        return self.projection_dim
    
    def get_projection_matrix(self) -> np.ndarray:
        """Return the underlying projection matrix."""
        return self.projection_matrix.copy()
    
    @classmethod
    def from_model(
        cls,
        model,
        projection_dim: int = 2,
        random_state: Optional[int] = None,
    ) -> "RandomWeightProjector":
        """Create projector from a PyTorch model.
        
        Args:
            model: PyTorch model to determine dimensionality
            projection_dim: Target dimension
            random_state: Random seed
            
        Returns:
            Initialized projector
        """
        # Compute total parameter count
        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        return cls(
            original_dim=total_params,
            projection_dim=projection_dim,
            random_state=random_state,
        )
    
    def save(self, path: str) -> None:
        """Save projection matrix to file."""
        np.savez_compressed(
            path,
            projection_matrix=self.projection_matrix,
            original_dim=self.original_dim,
            projection_dim=self.projection_dim,
        )
    
    @classmethod
    def load(cls, path: str) -> "RandomWeightProjector":
        """Load projection matrix from file."""
        data = np.load(path)
        projector = cls.__new__(cls)
        projector.projection_matrix = data["projection_matrix"]
        projector.original_dim = int(data["original_dim"])
        projector.projection_dim = int(data["projection_dim"])
        return projector


class IdentityProjector(WeightProjector):
    """Identity projection (no dimensionality reduction).
    
    Useful for testing or when projection is not needed.
    """
    
    def __init__(self):
        pass
    
    def project(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Return flattened weights without projection."""
        return self.flatten_weights(weights)
    
    def get_projection_dim(self) -> int:
        """Return None since dimensionality is variable."""
        return -1  # Indicates variable dimension