"""Node state management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class NodeState:
    """Encapsulates the mutable state of a node.
    
    This separates concerns and makes it easier to:
    - Serialize/checkpoint node state
    - Test state transitions
    - Implement stateful protocols
    """
    
    # Model parameters
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    projected_weights: Optional[np.ndarray] = None
    
    # Training metrics
    loss: float = 0.0
    accuracy: float = 0.0
    epoch: int = 0
    
    # Communication buffers
    buffer: List[Dict[str, np.ndarray]] = field(default_factory=list)
    buffer_projected: List[np.ndarray] = field(default_factory=list)
    
    # Consensus state
    last_consensus_round: int = 0
    consensus_coefficients: Optional[Dict[str, np.ndarray]] = None
    
    def reset_buffers(self) -> None:
        """Clear all communication buffers."""
        self.buffer.clear()
        self.buffer_projected.clear()
        self.consensus_coefficients = None
    
    def increment_epoch(self) -> None:
        """Increment the epoch counter."""
        self.epoch += 1
    
    def update_metrics(self, loss: float, accuracy: float) -> None:
        """Update training metrics."""
        self.loss = loss
        self.accuracy = accuracy
    
    def num_buffered_messages(self) -> int:
        """Return the number of messages in buffer."""
        return len(self.buffer)
    
    def has_projected_buffers(self) -> bool:
        """Check if projected buffers are available."""
        return len(self.buffer_projected) > 0
    
    def to_dict(self) -> Dict:
        """Serialize state to dictionary (for checkpointing)."""
        return {
            "weights": self.weights.copy(),
            "loss": self.loss,
            "accuracy": self.accuracy,
            "epoch": self.epoch,
            "last_consensus_round": self.last_consensus_round,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "NodeState":
        """Restore state from dictionary."""
        state = cls()
        state.weights = data.get("weights", {})
        state.loss = data.get("loss", 0.0)
        state.accuracy = data.get("accuracy", 0.0)
        state.epoch = data.get("epoch", 0)
        state.last_consensus_round = data.get("last_consensus_round", 0)
        return state