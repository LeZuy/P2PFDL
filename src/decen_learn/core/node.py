"""Honest node implementation for decentralized learning."""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

from .node_state import NodeState
from .weight_projector import WeightProjector
from .local_trainer import LocalTrainer
from .device_manager import DeviceManager
from .utils import flatten_weight_dict, unflatten_weight_dict

logger = logging.getLogger(__name__)


class Node:
    """Represents an honest participant in decentralized learning.
    
    Responsibilities:
    - Local training
    - Weight projection
    - Message passing (send/receive)
    - Consensus aggregation
    - Model updates
    """
    
    def __init__(
        self,
        node_id: int,
        model: torch.nn.Module,
        projector: WeightProjector,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ):
        """Initialize honest node.
        
        Args:
            node_id: Unique node identifier
            model: PyTorch model to train
            projector: Weight projection strategy
            dataloader: Training data
            device: Compute device (auto-assigned if None)
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization
        """
        self.id = node_id
        self.neighbors: List[int] = []
        
        # Core components
        self.model = model
        self.device_manager = DeviceManager(node_id, device)
        
        # Training
        device = self.device_manager.get_assigned_device()
        self.projector = projector.to(device)
        self.trainer = LocalTrainer(
            node_id = self.id,
            model=model,
            dataloader=dataloader,
            device=device,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.device_manager.track(self.model)
        
        # State
        self.state = NodeState()
        self._sync_weights_from_model()

    @staticmethod
    def _clone_weight_dict(
        weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return a detached clone of the provided weight dict."""
        return {
            name: tensor.detach().clone()
            for name, tensor in weights.items()
        }

    @staticmethod
    def _weights_to_numpy(
        weights: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Convert weight dict to CPU numpy arrays."""
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in weights.items()
        }

    @staticmethod
    def _weights_from_numpy(
        weights: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert numpy weight dict to torch tensors."""
        return {
            name: torch.as_tensor(values).clone()
            for name, values in weights.items()
        }
    
    @property
    def is_byzantine(self) -> bool:
        """Return True if this is a Byzantine (malicious) node."""
        return False
    
    @property
    def device(self) -> torch.device:
        """Return the assigned compute device."""
        return self.device_manager.get_assigned_device()
    
    # ========== Training ==========
    
    def train_epoch(self) -> Tuple[float, float]:
        """Execute one local training epoch.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        # allocated = torch.cuda.max_memory_allocated()
        # free = torch.cuda.memory_reserved()
        # print(f"[Node {self.id}] VRAM used = {(allocated)/1024**2:.1f} MB / {(free)/1024**2:.1f} MB", flush=True)
        # print(f"Memory snapshot : {torch.cuda.memory_snapshot()}", flush=True)
        # print(f"Device:{self.device}", flush=True)
        device = self.device_manager.acquire(self.model)
        
        loss, accuracy = self.trainer.train_epoch()
        
        self._sync_weights_from_model()
        self.state.update_metrics(loss, accuracy)
        self.state.increment_epoch()
        
        self.device_manager.release(self.model)
        # free, total = torch.cuda.mem_get_info()
        # print(f"[Node {self.id}] VRAM used = {(allocated)/1024**2:.1f} MB / {(free)/1024**2:.1f} MB", flush=True)
        # print(f"Memory snapshot : {torch.cuda.memory_snapshot()}", flush=True)
        # print(f"Device:{self.device}", flush=True)
        
        logger.debug(
            f"[Node {self.id}] Epoch {self.state.epoch}: "
            f"Loss={loss:.4f}, Acc={accuracy:.2f}%"
        )
        
        return loss, accuracy
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None
    ) -> Tuple[float, float]:
        """Evaluate model on test data.
        
        Args:
            dataloader: Test dataloader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        device = self.device_manager.acquire(self.model)
        loss, accuracy = self.trainer.evaluate(dataloader)
        self.device_manager.release(self.model)
        return loss, accuracy
    
    # ========== Communication ==========
    
    def prepare_broadcast(self) -> Dict[str, torch.Tensor]:
        """Prepare weights to send to neighbors.
        
        Returns:
            Deep copy of current weights
        """
        return self._clone_weight_dict(self.state.weights)
    
    def receive(self, weights: Dict[str, torch.Tensor]) -> None:
        """Receive weights from a neighbor.
        
        Args:
            weights: Weight dictionary from neighbor
        """
        self.state.buffer.append(self._clone_weight_dict(weights))
    
    def reset_buffers(self) -> None:
        """Clear all communication buffers."""
        self.state.reset_buffers()
    
    # ========== Consensus ==========
    
    def project_buffers(self) -> None:
        """Project all buffered weights to low-dimensional space."""
        projections = []
        for weights in self.state.buffer:
            projected = self.projector.project(weights)
            if not torch.is_tensor(projected):
                projected = torch.as_tensor(projected)
            projections.append(projected.detach().clone())
        self.state.buffer_projected = projections
    
    def aggregate(
        self,
        aggregator
    ) -> Dict[str, torch.Tensor]:
        """Aggregate buffered weights using provided aggregator.
        
        Args:
            aggregator: Aggregator instance (e.g., TverbergAggregator)
            
        Returns:
            Aggregated weights
        """
        if not self.state.buffer:
            logger.warning(f"[Node {self.id}] No buffered weights to aggregate")
            return self._clone_weight_dict(self.state.weights)
        
        use_projection = getattr(aggregator, "requires_projection", False)
        
        if use_projection:
            if not self.state.has_projected_buffers():
                self.project_buffers()
            vectors = torch.stack(self.state.buffer_projected, dim=0)
        else:
            vectors = torch.stack(
                [self._flatten_weights(w) for w in self.state.buffer],
                dim=0,
            )
        
        # Run aggregation
        result = aggregator(vectors)
        
        # Store projection result when applicable
        if use_projection:
            projected_vec = result.vector
            if not torch.is_tensor(projected_vec):
                projected_vec = torch.as_tensor(projected_vec)
            self.state.projected_weights = projected_vec.detach().cpu().clone()
        else:
            self.state.projected_weights = None
        
        # Store convex coefficients if available (for Tverberg)
        if result.weights is not None:
            coeffs = result.weights
            if isinstance(coeffs, torch.Tensor):
                coeffs = coeffs.detach()
            self.state.consensus_coefficients = {"__flat__": coeffs}
        
        # Return aggregated weights
        if result.selected_index is not None:
            # Selection-based (e.g., Krum)
            return self._clone_weight_dict(
                self.state.buffer[result.selected_index]
            )
        elif result.weights is not None:
            # Convex combination (e.g., Tverberg)
            return self._reconstruct_from_coefficients(result.weights)
        else:
            if use_projection:
                # Direct vector (e.g., Mean in projected space)
                return self._reconstruct_from_projection(result.vector)
            # Aggregated vector is already in the original weight space
            return self._unflatten_weights(result.vector)
    
    def _reconstruct_from_coefficients(
        self,
        coefficients,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct weights from convex combination coefficients.
        
        Args:
            coefficients: Convex weights (sum to 1)
            
        Returns:
            Reconstructed weight dictionary
        """
        # Stack buffered weights: shape (num_neighbors, total_params)
        stacked = torch.stack(
            [self._flatten_weights(w) for w in self.state.buffer],
            dim=0,
        )
        
        coeffs = coefficients
        if not torch.is_tensor(coeffs):
            coeffs = torch.as_tensor(
                coeffs,
                dtype=stacked.dtype,
                device=stacked.device,
            )
        flat_weights = coeffs @ stacked
        
        # Unflatten back to dictionary
        return self._unflatten_weights(flat_weights)
    
    def _reconstruct_from_projection(
        self,
        projected
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct weights from low-dimensional projection.
        
        This is an approximation since projection loses information.
        We return the convex combination that best matches the projection.
        
        Args:
            projected: Projected vector
            
        Returns:
            Reconstructed weight dictionary
        """
        # Find convex coefficients via RANSAC or Gilbert's algorithm
        from ..tverberg.ransac import ransac_simplex
        
        print(f"[Node {self.id}] Reconstructing from {len(self.state.buffer_projected)} vectors")
        projected_neighbors = torch.stack(
            [
                proj if torch.is_tensor(proj) else torch.as_tensor(proj)
                for proj in self.state.buffer_projected
            ],
            dim=0,
        ).detach().cpu().numpy()
        projected_np = (
            projected.detach().cpu().numpy()
            if torch.is_tensor(projected)
            else np.asarray(projected)
        )
        
        result = ransac_simplex(
            projected_neighbors,
            q=projected_np,
            mode="contain_q",
            iterations=10000,
            eps=1e-6,
        )
        
        if result["success"] and result["q_weights_dense"] is not None:
            coeffs = torch.as_tensor(
                result["q_weights_dense"],
                dtype=torch.float32,
            )
            return self._reconstruct_from_coefficients(coeffs)
        
        # Fallback: simple mean
        logger.warning(
            f"[Node {self.id}] RANSAC failed, using mean fallback"
        )
        return self._mean_of_buffers()
    
    def _mean_of_buffers(self) -> Dict[str, torch.Tensor]:
        """Compute mean of all buffered weights."""
        if not self.state.buffer:
            return self._clone_weight_dict(self.state.weights)
        
        # Initialize with zeros
        result = {
            name: torch.zeros_like(arr)
            for name, arr in self.state.buffer[0].items()
        }
        
        # Sum all buffers
        for weights in self.state.buffer:
            for name, arr in weights.items():
                result[name] += arr
        
        # Average
        n = len(self.state.buffer)
        for name in result:
            result[name] /= n
        
        return {name: tensor.clone() for name, tensor in result.items()}
    
    def update_weights(
        self,
        new_weights: Dict[str, torch.Tensor],
        _lambda: float = 0.0
    ) -> None:
        """Update model with new weights.
        
        Args:
            new_weights: New weight dictionary
            _lambda: Blending coefficient (0=full update, 1=no update)
        """
        if _lambda > 0:
            # Blend with current weights
            for name in new_weights:
                new_weights[name] = (
                    _lambda * self.state.weights[name] +
                    (1 - _lambda) * new_weights[name]
                )
        
        self.state.weights = self._clone_weight_dict(new_weights)
        self._sync_weights_to_model()
    
    # ========== Weight Management ==========
    
    def _sync_weights_from_model(self) -> None:
        """Extract weights from PyTorch model to state."""
        self.state.weights = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        projected = self.projector.project(self.state.weights)
        if not torch.is_tensor(projected):
            projected = torch.as_tensor(projected)
        self.state.projected_weights = projected.detach().cpu().clone()
    
    def _sync_weights_to_model(self) -> None:
        """Load weights from state into PyTorch model."""
        state_dict = self.model.state_dict()
        for name, values in self.state.weights.items():
            if name in state_dict:
                target = state_dict[name]
                reshaped = values.to(target.device).reshape(target.shape)
                target.copy_(reshaped)
    
    def _flatten_weights(
        self,
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Flatten weight dictionary to 1D vector."""
        return flatten_weight_dict(weights)
    
    def _unflatten_weights(
        self,
        flat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct weight dictionary from flat vector."""
        return unflatten_weight_dict(flat, self.state.weights)
    
    # ========== Utilities ==========
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
    
    def __repr__(self) -> str:
        return (
            f"Node(id={self.id}, "
            f"neighbors={len(self.neighbors)}, "
            f"byzantine={self.is_byzantine})"
        )
