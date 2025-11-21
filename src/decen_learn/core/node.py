# src/decen_learn/core/node.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol
import torch
import numpy as np

class WeightProjector(Protocol):
    """Protocol for weight projection strategies."""
    def project(self, weights: Dict[str, np.ndarray]) -> np.ndarray: ...
    def unproject(self, projected: np.ndarray, coefficients: np.ndarray) -> Dict[str, np.ndarray]: ...


@dataclass
class NodeState:
    """Encapsulates the mutable state of a node."""
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    projected_weights: Optional[np.ndarray] = None
    loss: float = 0.0
    buffer: List[Dict[str, np.ndarray]] = field(default_factory=list)
    buffer_projected: List[np.ndarray] = field(default_factory=list)


class Node:
    """Represents a participant in decentralized learning."""
    
    def __init__(
        self,
        node_id: int,
        model: torch.nn.Module,
        projector: WeightProjector,
        config: "NodeConfig",
        device: Optional[torch.device] = None,
    ):
        self.id = node_id
        self.config = config
        self.device = device or self._select_device(node_id)
        
        self._model = model.to(self.device)
        self._projector = projector
        self._trainer = LocalTrainer(model, config.training, self.device)
        
        self.neighbors: List[int] = []
        self.state = NodeState()
        self._sync_weights_from_model()
    
    @property
    def is_byzantine(self) -> bool:
        return False
    
    def train_epoch(self) -> float:
        """Run one local training epoch."""
        loss = self._trainer.train_epoch()
        self._sync_weights_from_model()
        self.state.loss = loss
        return loss
    
    def prepare_broadcast(self) -> Dict[str, np.ndarray]:
        """Prepare weights to send to neighbors."""
        return self.state.weights.copy()
    
    def receive(self, weights: Dict[str, np.ndarray]) -> None:
        """Receive weights from a neighbor."""
        self.state.buffer.append(weights)
    
    def aggregate(self, aggregator: "BaseAggregator") -> Dict[str, np.ndarray]:
        """Aggregate buffered weights using the given aggregator."""
        if not self.state.buffer:
            return self.state.weights
        
        # Project all buffered weights
        projected = np.stack([
            self._projector.project(w) for w in self.state.buffer
        ])
        
        result = aggregator(projected)
        
        # Recover full weights from projection
        if result.weights is not None:
            return self._projector.unproject(result.vector, result.weights)
        
        # Fallback: use the selected index
        if result.selected_index is not None:
            return self.state.buffer[result.selected_index]
        
        raise ValueError("Aggregator must return weights or selected_index")
    
    def update_weights(self, new_weights: Dict[str, np.ndarray], momentum: float = 0.0) -> None:
        """Update model with new weights."""
        if momentum > 0:
            for name in new_weights:
                new_weights[name] = (
                    momentum * self.state.weights[name] + 
                    (1 - momentum) * new_weights[name]
                )
        
        self.state.weights = new_weights
        self._sync_weights_to_model()
    
    def reset_buffer(self) -> None:
        """Clear communication buffers."""
        self.state.buffer.clear()
        self.state.buffer_projected.clear()
    
    def _sync_weights_from_model(self) -> None:
        """Extract weights from PyTorch model."""
        self.state.weights = {
            name: param.detach().cpu().numpy().copy()
            for name, param in self._model.named_parameters()
        }
        self.state.projected_weights = self._projector.project(self.state.weights)
    
    def _sync_weights_to_model(self) -> None:
        """Load weights into PyTorch model."""
        state_dict = self._model.state_dict()
        for name, values in self.state.weights.items():
            if name in state_dict:
                state_dict[name] = torch.from_numpy(values).reshape(state_dict[name].shape)
        self._model.load_state_dict(state_dict)
    
    @staticmethod
    def _select_device(node_id: int) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{node_id % torch.cuda.device_count()}")
        return torch.device("cpu")
