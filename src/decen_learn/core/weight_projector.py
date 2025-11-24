"""Weight projection strategies for dimensionality reduction."""

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import torch

from .utils import flatten_weight_dict


class WeightProjector(ABC):
    """Protocol for weight projection strategies."""
    
    @abstractmethod
    def project(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project high-dimensional weights to low-dimensional space.
        
        Args:
            weights: Dictionary {layer: weight arrays}
            
        Returns:
            Low-dimensional projection vector
        """
        pass
    
    @abstractmethod
    def get_projection_dim(self) -> int:
        """Return the dimensionality of the projected space."""
        pass
    
    def flatten_weights(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten weight dictionary to 1D vector."""
        return flatten_weight_dict(
            weights,
            detach=True,
            to_device=torch.device("cpu"),
        )


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
        device: Optional[torch.device] = None,
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
        self.dtype = torch.float32
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        
        # Build projection matrix directly in torch for downstream tensor ops
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(random_state)
        self.projection_matrix = torch.randn(
            projection_dim,
            original_dim,
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        ) / math.sqrt(projection_dim)
        self._matrix_cache = {self.device: self.projection_matrix}
    
    def project(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project weights to low-dimensional space.
        
        Args:
            weights: Weight dictionary
            
        Returns:
            Projected vector of shape (projection_dim,)
        """
        flat = self.flatten_weights(weights)
        
        if flat.numel() != self.original_dim:
            raise ValueError(
                f"Weight dimension mismatch: expected {self.original_dim}, "
                f"got {flat.numel()}"
            )
        
        # Project: result = P @ x where P is (proj_dim, orig_dim)
        matrix = self._get_projection_matrix(self.device)
        projected = torch.matmul(matrix, flat)
        return projected
    
    def flatten_weights(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten weight dictionary to 1D vector on projector device."""
        flat = flatten_weight_dict(
            weights,
            detach=True,
            to_device=self.device,
        )
        return flat.to(self.dtype)
    
    def get_projection_dim(self) -> int:
        """Return projection dimensionality."""
        return self.projection_dim
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Return the underlying projection matrix."""
        return self.projection_matrix.clone()
    
    def to(self, device: torch.device) -> "RandomWeightProjector":
        """Return a copy of this projector on the specified device."""
        target = torch.device(device)
        if target == self.device:
            return self
        new_proj = self.__class__.__new__(self.__class__)
        new_proj.original_dim = self.original_dim
        new_proj.projection_dim = self.projection_dim
        new_proj.random_state = self.random_state
        new_proj.dtype = self.dtype
        new_proj.device = target
        new_proj.projection_matrix = self.projection_matrix.to(target)
        new_proj._matrix_cache = {target: new_proj.projection_matrix}
        return new_proj
    
    def _get_projection_matrix(self, device: torch.device) -> torch.Tensor:
        """Return projection matrix on requested device."""
        if device not in self._matrix_cache:
            self._matrix_cache[device] = self.projection_matrix.to(device)
        return self._matrix_cache[device]
    
    @classmethod
    def from_model(
        cls,
        model,
        projection_dim: int = 2,
        random_state: Optional[int] = None,
        device: Optional[torch.device] = None,
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
            device=device,
        )
    
    def save(self, path: str) -> None:
        """Save projection matrix to file."""
        np.savez_compressed(
            path,
            projection_matrix=self.projection_matrix.detach().cpu().numpy(),
            original_dim=self.original_dim,
            projection_dim=self.projection_dim,
        )
    
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None,
    ) -> "RandomWeightProjector":
        """Load projection matrix from file."""
        data = np.load(path)
        projector = cls.__new__(cls)
        matrix = torch.as_tensor(data["projection_matrix"]).detach().clone()
        projector.original_dim = int(data["original_dim"])
        projector.projection_dim = int(data["projection_dim"])
        projector.random_state = None
        projector.dtype = matrix.dtype
        projector.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        projector.projection_matrix = matrix.to(projector.device)
        projector._matrix_cache = {projector.device: projector.projection_matrix}
        return projector


class IdentityProjector(WeightProjector):
    """Identity projection (no dimensionality reduction).
    
    Useful for testing or when projection is not needed.
    """
    
    def __init__(self):
        pass
    
    def project(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return flattened weights without projection."""
        return self.flatten_weights(weights)
    
    def get_projection_dim(self) -> int:
        """Return None since dimensionality is variable."""
        return -1  # Indicates variable dimension
