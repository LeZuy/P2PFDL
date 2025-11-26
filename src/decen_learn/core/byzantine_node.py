"""Byzantine (malicious) node implementation."""

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
import logging

from .node import Node
from .weight_projector import WeightProjector

logger = logging.getLogger(__name__)


class ByzantineNode(Node):
    """A malicious node that crafts adversarial updates.
    
    Key differences from honest nodes:
    - Crafts malicious broadcasts using attack strategy
    - Ignores consensus updates
    - May train on poisoned data
    """
    
    def __init__(
        self,
        node_id: int,
        model: torch.nn.Module,
        projector: WeightProjector,
        dataloader: DataLoader,
        attack,  # BaseAttack instance
        bad_client_ids: List[int],
        device: Optional[torch.device] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        boosting_factor: float = 1.0,
    ):
        """Initialize Byzantine node.
        
        Args:
            node_id: Unique node identifier
            model: PyTorch model
            projector: Weight projector
            dataloader: Training data (possibly poisoned)
            attack: Attack strategy instance
            bad_client_ids: List of all Byzantine node IDs
            device: Compute device
            learning_rate: Learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization
            boosting_factor: Attack amplification factor
        """
        super().__init__(
            node_id=node_id,
            model=model,
            projector=projector,
            dataloader=dataloader,
            device=device,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        
        device_assigned = self.device
        attack.set_device(device_assigned)
        self._attack = attack
        self._bad_client_ids = set(bad_client_ids)
        self._boosting_factor = boosting_factor
        logger.info(
            f"[Node {self.id}] Initialized as BYZANTINE node with "
            f"{attack.__class__.__name__}"
        )
    
    @property
    def is_byzantine(self) -> bool:
        """Return True to identify as Byzantine."""
        return True
    
    # ========== Malicious Behavior ==========
    
    def prepare_broadcast(
        self,
        processes: Optional[List[Node]] = None
    ) -> Dict[str, torch.Tensor]:
        """Craft malicious weights to broadcast to neighbors.
        
        Args:
            processes: All nodes in the network (to access neighbor states)
            
        Returns:
            Malicious weight dictionary
        """
        # Collect honest neighbor weights for attack computation
        honest_weights = []
        
        if processes is not None:
            for neighbor_id in self.neighbors:
                neighbor = processes[neighbor_id]
                if not neighbor.is_byzantine:
                    honest_weights.append(neighbor.state.weights)
        
        # If no honest neighbors or no access to processes, use buffered weights
        if not honest_weights and self.state.buffer:
            honest_weights = [w for w in self.state.buffer]
        
        # Fallback: return own weights if no information available
        if not honest_weights:
            logger.warning(
                f"[Byzantine {self.id}] No honest weights available, "
                "returning own weights"
            )
            return self._clone_weight_dict(self.state.weights)
        
        # Craft malicious weights using attack strategy
        try:
            malicious_weights = self._attack.craft(
                honest_weights=honest_weights,
                attacker_weights=self.state.weights,
            )
            
            # Apply boosting factor
            if self._boosting_factor != 1.0:
                malicious_weights = self._boost_weights(
                    malicious_weights,
                    self._boosting_factor
                )
            
            logger.debug(
                f"[Byzantine {self.id}] Crafted malicious update using "
                f"{self._attack.__class__.__name__}"
            )
            
            return malicious_weights
        
        except Exception as e:
            logger.error(
                f"[Byzantine {self.id}] Attack crafting failed: {e}. "
                "Returning own weights."
            )
            return self._clone_weight_dict(self.state.weights)
    
    def _boost_weights(
        self,
        weights: Dict[str, torch.Tensor],
        factor: float
    ) -> Dict[str, torch.Tensor]:
        """Apply boosting factor to weights.
        
        Args:
            weights: Weight dictionary
            factor: Boosting factor
            
        Returns:
            Boosted weights
        """
        return {
            name: tensor * factor
            for name, tensor in weights.items()
        }
    
    def receive(self, weights: Dict[str,  torch.Tensor]) -> None:
        """Receive weights from a neighbor.
        
        Byzantine nodes may selectively ignore certain messages.
        For now, we accept all messages like honest nodes.
        """
        super().receive(weights)
    
    def update_weights(
        self,
        new_weights: Dict[str, torch.Tensor],
        momentum: float = 0.0
    ) -> None:
        """Byzantine nodes ignore consensus updates to preserve malicious state.
        
        This is a key difference from honest nodes - Byzantine nodes
        don't update based on consensus results.
        """
        logger.debug(
            f"[Byzantine {self.id}] Ignoring consensus update "
            "(maintaining malicious state)"
        )
        # Intentionally do nothing
        pass
    
    def aggregate(self, aggregator) -> Dict[str, torch.Tensor]:
        """Byzantine nodes don't aggregate - they return their own weights.
        
        Args:
            aggregator: Ignored
            
        Returns:
            Own weights
        """
        return self._clone_weight_dict(self.state.weights)
    
    # ========== Training (Optional Poisoning) ==========
    
    def train_epoch(self) -> tuple[float, float]:
        """Train with potentially poisoned data.
        
        Byzantine nodes can still train locally to maintain credibility
        or to learn poisoned patterns.
        """
        # Optional: Byzantine nodes might skip training entirely
        # For now, train normally but on potentially poisoned data
        return super().train_epoch()
    
    def __repr__(self) -> str:
        return (
            f"ByzantineNode(id={self.id}, "
            f"attack={self._attack.__class__.__name__}, "
            f"neighbors={len(self.neighbors)})"
        )
