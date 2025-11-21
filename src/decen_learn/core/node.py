"""Honest node implementation for decentralized learning."""

import copy
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

from .node_state import NodeState
from .weight_projector import WeightProjector
from .local_trainer import LocalTrainer
from .device_manager import DeviceManager

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
        self.projector = projector
        self.device_manager = DeviceManager(node_id, device)
        
        # Training
        device = self.device_manager.get_assigned_device()
        self.trainer = LocalTrainer(
            model=model,
            dataloader=dataloader,
            device=device,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        
        # State
        self.state = NodeState()
        self._sync_weights_from_model()
    
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
        device = self.device_manager.acquire(self.model)
        
        loss, accuracy = self.trainer.train_epoch()
        
        self._sync_weights_from_model()
        self.state.update_metrics(loss, accuracy)
        self.state.increment_epoch()
        
        self.device_manager.release(self.model)
        
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
    
    def prepare_broadcast(self) -> Dict[str, np.ndarray]:
        """Prepare weights to send to neighbors.
        
        Returns:
            Deep copy of current weights
        """
        return copy.deepcopy(self.state.weights)
    
    def receive(self, weights: Dict[str, np.ndarray]) -> None:
        """Receive weights from a neighbor.
        
        Args:
            weights: Weight dictionary from neighbor
        """
        self.state.buffer.append(copy.deepcopy(weights))
    
    def reset_buffers(self) -> None:
        """Clear all communication buffers."""
        self.state.reset_buffers()
    
    # ========== Consensus ==========
    
    def project_buffers(self) -> None:
        """Project all buffered weights to low-dimensional space."""
        self.state.buffer_projected = [
            self.projector.project(weights)
            for weights in self.state.buffer
        ]
    
    def aggregate(
        self,
        aggregator
    ) -> Dict[str, np.ndarray]:
        """Aggregate buffered weights using provided aggregator.
        
        Args:
            aggregator: Aggregator instance (e.g., KrumAggregator)
            
        Returns:
            Aggregated weights
        """
        if not self.state.buffer:
            logger.warning(f"[Node {self.id}] No buffered weights to aggregate")
            return copy.deepcopy(self.state.weights)
        
        # Use projected buffers if available, otherwise project now
        if not self.state.has_projected_buffers():
            self.project_buffers()
        
        # Stack projected vectors
        projected = np.stack(self.state.buffer_projected, axis=0)
        
        # Run aggregation
        result = aggregator(projected)
        
        # Store projection result
        self.state.projected_weights = result.vector
        
        # Store convex coefficients if available (for Tverberg)
        if result.weights is not None:
            self.state.consensus_coefficients = {"__flat__": result.weights}
        
        # Return aggregated weights
        if result.selected_index is not None:
            # Selection-based (e.g., Krum)
            return copy.deepcopy(self.state.buffer[result.selected_index])
        elif result.weights is not None:
            # Convex combination (e.g., Tverberg)
            return self._reconstruct_from_coefficients(result.weights)
        else:
            # Direct vector (e.g., Mean in projected space)
            return self._reconstruct_from_projection(result.vector)
    
    def _reconstruct_from_coefficients(
        self,
        coefficients: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Reconstruct weights from convex combination coefficients.
        
        Args:
            coefficients: Convex weights (sum to 1)
            
        Returns:
            Reconstructed weight dictionary
        """
        # Stack buffered weights: shape (num_neighbors, total_params)
        stacked = np.stack([
            self._flatten_weights(w) for w in self.state.buffer
        ], axis=0)
        
        # Weighted sum: alpha @ stacked
        flat_weights = coefficients @ stacked
        
        # Unflatten back to dictionary
        return self._unflatten_weights(flat_weights)
    
    def _reconstruct_from_projection(
        self,
        projected: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Reconstruct weights from low-dimensional projection.
        
        This is an approximation since projection loses information.
        We return the convex combination that best matches the projection.
        
        Args:
            projected: Projected vector
            
        Returns:
            Reconstructed weight dictionary
        """
        # Find convex coefficients via RANSAC or Gilbert's algorithm
        from src.decen_learn.projection.ransac import ransac_simplex
        
        projected_neighbors = np.stack(self.state.buffer_projected, axis=0)
        
        result = ransac_simplex(
            projected_neighbors,
            q=projected,
            mode="contain_q",
            iterations=10000,
            eps=1e-9,
        )
        
        if result["success"] and result["q_weights_dense"] is not None:
            return self._reconstruct_from_coefficients(
                result["q_weights_dense"]
            )
        
        # Fallback: simple mean
        logger.warning(
            f"[Node {self.id}] RANSAC failed, using mean fallback"
        )
        return self._mean_of_buffers()
    
    def _mean_of_buffers(self) -> Dict[str, np.ndarray]:
        """Compute mean of all buffered weights."""
        if not self.state.buffer:
            return copy.deepcopy(self.state.weights)
        
        # Initialize with zeros
        result = {
            name: np.zeros_like(arr)
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
        
        return result
    
    def update_weights(
        self,
        new_weights: Dict[str, np.ndarray],
        momentum: float = 0.0
    ) -> None:
        """Update model with new weights.
        
        Args:
            new_weights: New weight dictionary
            momentum: Mixing coefficient (0=full update, 1=no update)
        """
        if momentum > 0:
            # Blend with current weights
            for name in new_weights:
                new_weights[name] = (
                    momentum * self.state.weights[name] +
                    (1 - momentum) * new_weights[name]
                )
        
        self.state.weights = new_weights
        self._sync_weights_to_model()
    
    # ========== Weight Management ==========
    
    def _sync_weights_from_model(self) -> None:
        """Extract weights from PyTorch model to state."""
        self.state.weights = {
            name: param.detach().cpu().numpy().copy()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.state.projected_weights = self.projector.project(
            self.state.weights
        )
    
    def _sync_weights_to_model(self) -> None:
        """Load weights from state into PyTorch model."""
        state_dict = self.model.state_dict()
        for name, values in self.state.weights.items():
            if name in state_dict:
                tensor = torch.from_numpy(values).reshape(
                    state_dict[name].shape
                )
                state_dict[name].copy_(tensor)
    
    def _flatten_weights(
        self,
        weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Flatten weight dictionary to 1D vector."""
        return np.concatenate([w.flatten() for w in weights.values()])
    
    def _unflatten_weights(
        self,
        flat: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Reconstruct weight dictionary from flat vector."""
        result = {}
        pos = 0
        for name, arr in self.state.weights.items():
            size = arr.size
            result[name] = flat[pos:pos + size].reshape(arr.shape).copy()
            pos += size
        return result
    
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