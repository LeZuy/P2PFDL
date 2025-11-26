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
            projected = self.projector.project_layerwise(weights)
            layer_dict: Dict[str, torch.Tensor] = {}
            for name, tensor in projected.items():
                proj_tensor = tensor
                if not torch.is_tensor(proj_tensor):
                    proj_tensor = torch.as_tensor(proj_tensor)
                layer_dict[name] = proj_tensor.detach().clone()
            projections.append(layer_dict)
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
        
        layer_names = list(self.state.weights.keys())
        layer_vectors = {
            name: self._stack_layer(name)
            for name in layer_names
        }
        if not self.state.has_projected_buffers():
            self.project_buffers()
        if not self.state.buffer_projected:
            raise RuntimeError(
                f"[Node {self.id}] Failed to build projected buffers for aggregation"
            )
        projected_vectors = {
            name: self._stack_projected_layer(name)
            for name in layer_names
        }
        
        aggregated: Dict[str, torch.Tensor] = {}
        projected_summary: Dict[str, torch.Tensor] = {}
        layer_coefficients: Dict[str, torch.Tensor] = {}
        for name in layer_names:
            inputs = projected_vectors[name]
            result = aggregator(inputs)
            flat_tensor, coeffs_tensor, proj_vector = self._process_layer_result(
                layer_name=name,
                result=result,
                layer_stack=layer_vectors[name],
                projected_stack=projected_vectors[name],
            )
            aggregated[name] = flat_tensor.reshape_as(self.state.weights[name])
            if coeffs_tensor is not None:
                layer_coefficients[name] = coeffs_tensor.detach().clone()
            if proj_vector is not None:
                projected_summary[name] = proj_vector.detach().cpu().clone()
        
        self.state.projected_weights = projected_summary or None
        self.state.consensus_coefficients = (
            layer_coefficients if layer_coefficients else None
        )
        
        return {name: tensor.clone() for name, tensor in aggregated.items()}
    
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
        if isinstance(coefficients, dict):
            return self._reconstruct_from_layerwise_coefficients(coefficients)
        
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

    def _reconstruct_from_layerwise_coefficients(
        self,
        coefficients: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct each layer from its convex coefficients."""
        aggregated: Dict[str, torch.Tensor] = {}
        for name, coeffs in coefficients.items():
            layer_stack = self._stack_layer(name)
            coeff_tensor = (
                coeffs if torch.is_tensor(coeffs) else torch.as_tensor(coeffs)
            )
            coeff_tensor = coeff_tensor.to(layer_stack.device, layer_stack.dtype)
            aggregated[name] = (coeff_tensor @ layer_stack).reshape_as(
                self.state.weights[name]
            )
        return {name: tensor.clone() for name, tensor in aggregated.items()}

    def _process_layer_result(
        self,
        *,
        layer_name: str,
        result,
        layer_stack: torch.Tensor,
        projected_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert projection-space aggregator output into flattened weights."""
        coeffs_tensor: Optional[torch.Tensor] = None
        projection_vector: Optional[torch.Tensor] = None
        if result.selected_index is not None:
            idx = int(result.selected_index)
            flat = layer_stack[idx]
            projection_vector = projected_stack[idx]
        elif result.weights is not None:
            coeffs_tensor = (
                result.weights
                if torch.is_tensor(result.weights)
                else torch.as_tensor(result.weights)
            )
            coeffs_tensor = coeffs_tensor.to(layer_stack.device, layer_stack.dtype)
            flat = coeffs_tensor @ layer_stack
            projection_vector = coeffs_tensor @ projected_stack
        else:
            projection_vector = result.vector
            if projection_vector is None:
                raise RuntimeError(
                    f"Aggregator did not return projection for layer '{layer_name}'"
                )
            if not torch.is_tensor(projection_vector):
                projection_vector = torch.as_tensor(projection_vector)
            flat, coeffs_tensor = self._reconstruct_layer_from_projection(
                layer_name=layer_name,
                projected_neighbors=projected_stack,
                target_projection=projection_vector,
                layer_stack=layer_stack,
            )
        return flat.reshape(-1), coeffs_tensor, projection_vector

    def _reconstruct_layer_from_projection(
        self,
        *,
        layer_name: str,
        projected_neighbors: torch.Tensor,
        target_projection: torch.Tensor,
        layer_stack: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Recover high-dimensional layer weights from projection output."""
        from ..tverberg.ransac import ransac_simplex
        
        neighbors_np = projected_neighbors.detach().cpu().numpy()
        target_np = (
            target_projection.detach().cpu().numpy()
            if torch.is_tensor(target_projection)
            else np.asarray(target_projection)
        )
        result = ransac_simplex(
            neighbors_np,
            q=target_np,
            mode="contain_q",
            iterations=10000,
            eps=1e-6,
        )
        if result["success"] and result["q_weights_dense"] is not None:
            coeffs = torch.as_tensor(
                result["q_weights_dense"],
                dtype=layer_stack.dtype,
                device=layer_stack.device,
            )
            flat = coeffs @ layer_stack
            return flat, coeffs
        logger.warning(
            f"[Node {self.id}] RANSAC failed for layer '{layer_name}', using mean fallback"
        )
        return layer_stack.mean(dim=0), None
    
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
        blending_lambda: float = 0.0
    ) -> None:
        """Update model with new weights.
        
        Args:
            new_weights: New weight dictionary
            blending_lambda: Blending coefficient (0=full update, 1=no update)
        """
        if blending_lambda > 0:
            # Blend with current weights
            for name in new_weights:
                new_weights[name] = (
                    blending_lambda * self.state.weights[name] +
                    (1 - blending_lambda) * new_weights[name]
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
        projected = self.projector.project_layerwise(self.state.weights)
        self.state.projected_weights = {
            name: (
                tensor.detach().cpu().clone()
                if torch.is_tensor(tensor)
                else torch.as_tensor(tensor).detach().cpu().clone()
            )
            for name, tensor in projected.items()
        }
    
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
    
    def _stack_layer(self, layer_name: str) -> torch.Tensor:
        """Stack buffered weights for a specific layer."""
        return torch.stack(
            [
                weights[layer_name].reshape(-1)
                for weights in self.state.buffer
            ],
            dim=0,
        )
    
    def _stack_projected_layer(self, layer_name: str) -> torch.Tensor:
        """Stack projected buffers for a specific layer."""
        return torch.stack(
            [
                proj[layer_name]
                if torch.is_tensor(proj[layer_name])
                else torch.as_tensor(proj[layer_name])
                for proj in self.state.buffer_projected
            ],
            dim=0,
        )
    
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
