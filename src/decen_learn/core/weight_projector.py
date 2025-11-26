"""Weight projection strategies for dimensionality reduction."""

import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple

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

    @abstractmethod
    def project_layerwise(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project each layer separately.
        
        Args:
            weights: Mapping of layer name to parameter tensor.
        
        Returns:
            Dictionary of layer name to low-dimensional projection.
        """
        raise NotImplementedError
    
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
        self.layer_order: Optional[List[str]] = None
        self.layer_slices: Optional[Dict[str, Tuple[int, int]]] = None
        self._layer_name_set: Optional[set] = None
        self.projection_matrix: Optional[torch.Tensor] = None
        self._matrix_cache: Dict[torch.device, torch.Tensor] = {}
        self.layer_projection_matrices: Optional[Dict[str, torch.Tensor]] = None
        self._layer_matrix_cache: Dict[torch.device, Dict[str, torch.Tensor]] = {}
    
    def project(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project weights to low-dimensional space.
        
        Args:
            weights: Weight dictionary
            
        Returns:
            Projected vector of shape (projection_dim,)
        """
        self._ensure_layer_metadata(weights)
        self._ensure_layer_projection_matrices()
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

    def project_layerwise(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Project each layer independently using consistent slices."""
        self._ensure_layer_metadata(weights)
        self._ensure_layer_projection_matrices()
        projections: Dict[str, torch.Tensor] = {}
        if not self.layer_slices:
            return projections
        ordered = self._ordered_weights(weights)
        for name in self.layer_order or []:
            tensor = ordered[name]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            layer_flat = tensor.detach().to(self.device).reshape(-1).to(self.dtype)
            start, end = self.layer_slices[name]
            layer_size = end - start
            if int(layer_flat.numel()) != layer_size:
                raise ValueError(
                    f"Layer '{name}' size mismatch: expected {layer_size}, "
                    f"got {layer_flat.numel()}"
                )
            matrix_slice = self._get_layer_matrix(name, self.device)
            if layer_size == 0:
                projections[name] = torch.zeros(
                    self.projection_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                projections[name] = torch.matmul(matrix_slice, layer_flat)
        return projections
    
    def flatten_weights(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten weight dictionary to 1D vector on projector device."""
        ordered = self._ordered_weights(weights)
        flat = flatten_weight_dict(
            ordered,
            detach=True,
            to_device=self.device,
        )
        return flat.to(self.dtype)
    
    def get_projection_dim(self) -> int:
        """Return projection dimensionality."""
        return self.projection_dim
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Return the underlying projection matrix."""
        if self.projection_matrix is None:
            raise RuntimeError(
                "Projection matrix is not initialized. "
                "Call project() or provide weights to build it."
            )
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
        if self.projection_matrix is not None:
            new_proj.projection_matrix = self.projection_matrix.to(target)
            new_proj._matrix_cache = {target: new_proj.projection_matrix}
        else:
            new_proj.projection_matrix = None
            new_proj._matrix_cache = {}
        if self.layer_projection_matrices is not None:
            new_proj.layer_projection_matrices = {
                name: matrix.to(target)
                for name, matrix in self.layer_projection_matrices.items()
            }
            new_proj._layer_matrix_cache = {
                target: new_proj.layer_projection_matrices
            }
        else:
            new_proj.layer_projection_matrices = None
            new_proj._layer_matrix_cache = {}
        if self.layer_order is not None:
            new_proj.layer_order = list(self.layer_order)
        else:
            new_proj.layer_order = None
        if self.layer_slices is not None:
            new_proj.layer_slices = {
                name: (start, end)
                for name, (start, end) in self.layer_slices.items()
            }
        else:
            new_proj.layer_slices = None
        if self._layer_name_set is not None:
            new_proj._layer_name_set = set(self._layer_name_set)
        else:
            new_proj._layer_name_set = None
        return new_proj
    
    def _get_projection_matrix(self, device: torch.device) -> torch.Tensor:
        """Return projection matrix on requested device."""
        if self.projection_matrix is None:
            raise RuntimeError(
                "Projection matrix requested before initialization."
            )
        if device not in self._matrix_cache:
            self._matrix_cache[device] = self.projection_matrix.to(device)
        return self._matrix_cache[device]
    
    def _get_layer_matrix(
        self,
        layer_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the projection block for a specific layer."""
        if self.layer_projection_matrices is None:
            raise RuntimeError(
                "Layer projection matrices are not initialized."
            )
        target = torch.device(device)
        if target not in self._layer_matrix_cache:
            self._layer_matrix_cache[target] = {
                name: matrix.to(target)
                for name, matrix in self.layer_projection_matrices.items()
            }
        return self._layer_matrix_cache[target][layer_name]
    
    def _ensure_layer_projection_matrices(self) -> None:
        """Build per-layer projection matrices if missing."""
        if self.layer_projection_matrices is not None:
            return
        if not self.layer_slices or not self.layer_order:
            return
        self._initialize_layer_projection_matrices()
    
    def _initialize_layer_projection_matrices(self) -> None:
        """Initialize per-layer projection matrices."""
        if self.layer_projection_matrices is not None:
            return
        if not self.layer_order or not self.layer_slices:
            return
        if self.projection_matrix is not None:
            base = self.projection_matrix.to(self.device)
            self.projection_matrix = base
            matrices: Dict[str, torch.Tensor] = {}
            for name in self.layer_order:
                start, end = self.layer_slices[name]
                matrices[name] = base[:, start:end].clone()
            self.layer_projection_matrices = matrices
            self._layer_matrix_cache = {self.device: matrices}
            self._matrix_cache = {self.device: base}
            return
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.random_state)
        matrices = {}
        for name in self.layer_order:
            start, end = self.layer_slices[name]
            layer_size = end - start
            if layer_size == 0:
                matrices[name] = torch.zeros(
                    self.projection_dim,
                    0,
                    dtype=self.dtype,
                    device=self.device,
                )
                continue
            matrices[name] = torch.randn(
                self.projection_dim,
                layer_size,
                generator=generator,
                dtype=self.dtype,
                device=self.device,
            ) / math.sqrt(self.projection_dim)
        if self.layer_order:
            concatenated = torch.cat(
                [matrices[name] for name in self.layer_order],
                dim=1,
            )
        else:
            concatenated = torch.empty(
                self.projection_dim,
                0,
                dtype=self.dtype,
                device=self.device,
            )
        self.layer_projection_matrices = matrices
        self.projection_matrix = concatenated
        self._layer_matrix_cache = {self.device: matrices}
        self._matrix_cache = {self.device: self.projection_matrix}
    
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
        
        projector = cls(
            original_dim=total_params,
            projection_dim=projection_dim,
            random_state=random_state,
            device=device,
        )
        weights = OrderedDict(
            (
                name,
                param.detach().clone(),
            )
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        if weights:
            projector._capture_layer_metadata(weights)
        return projector
    
    def save(self, path: str) -> None:
        """Save projection matrix to file."""
        if self.projection_matrix is None:
            raise RuntimeError(
                "Projection matrix has not been initialized; cannot save."
            )
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
        projector.layer_order = None
        projector.layer_slices = None
        projector._layer_name_set = None
        projector.layer_projection_matrices = None
        projector._layer_matrix_cache = {}
        return projector

    def _ordered_weights(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return weights ordered consistently with stored metadata."""
        self._ensure_layer_metadata(weights)
        if not self.layer_order:
            return weights
        ordered = OrderedDict()
        for name in self.layer_order:
            ordered[name] = weights[name]
        return ordered
    
    def _ensure_layer_metadata(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        if self.layer_order is None or self.layer_slices is None:
            self._capture_layer_metadata(weights)
        else:
            self._validate_layer_metadata(weights)
    
    def _capture_layer_metadata(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        order: List[str] = []
        slices: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for name, tensor in weights.items():
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            size = int(tensor.numel())
            order.append(name)
            slices[name] = (offset, offset + size)
            offset += size
        if offset != self.original_dim:
            raise ValueError(
                "Layer sizes do not sum to original dimension: "
                f"got {offset}, expected {self.original_dim}"
            )
        self.layer_order = order
        self.layer_slices = slices
        self._layer_name_set = set(order)
        # Reset cached projections to rebuild with the new layout
        self.layer_projection_matrices = None
        self.projection_matrix = None
        self._matrix_cache = {}
        self._layer_matrix_cache = {}
        self._initialize_layer_projection_matrices()
    
    def _validate_layer_metadata(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        if self.layer_order is None or self.layer_slices is None:
            return self._capture_layer_metadata(weights)
        incoming = set(weights.keys())
        if self._layer_name_set is None:
            self._layer_name_set = set(self.layer_order)
        if incoming != self._layer_name_set:
            missing = self._layer_name_set - incoming
            extra = incoming - self._layer_name_set
            raise ValueError(
                "Inconsistent layer keys for projection: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name in self.layer_order:
            start, end = self.layer_slices[name]
            expected = end - start
            tensor = weights[name]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            actual = int(tensor.numel())
            if expected != actual:
                raise ValueError(
                    f"Layer '{name}' size changed (expected {expected}, got {actual})"
                )


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

    def project_layerwise(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return a flattened copy of each layer."""
        projections: Dict[str, torch.Tensor] = {}
        for name, tensor in weights.items():
            layer_tensor = tensor
            if not torch.is_tensor(layer_tensor):
                layer_tensor = torch.as_tensor(layer_tensor)
            projections[name] = layer_tensor.detach().reshape(-1).clone()
        return projections
