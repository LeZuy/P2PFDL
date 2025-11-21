# src/decen_learn/training/trainer.py
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

@dataclass
class EpochMetrics:
    """Metrics collected during one epoch."""
    epoch: int
    losses: np.ndarray
    accuracies: Optional[np.ndarray] = None
    consensus_time: float = 0.0


class DecentralizedTrainer:
    """Orchestrates the decentralized training process."""
    
    def __init__(
        self,
        nodes: List["Node"],
        aggregator: "BaseAggregator",
        config: "TrainingConfig",
        callbacks: Optional[List["TrainingCallback"]] = None,
    ):
        self.nodes = nodes
        self.aggregator = aggregator
        self.config = config
        self.callbacks = callbacks or []
        
        self._topology = None
        self._metrics_history: List[EpochMetrics] = []
    
    def set_topology(self, adjacency_matrix: np.ndarray) -> None:
        """Configure node neighborhoods from adjacency matrix."""
        for node in self.nodes:
            node.neighbors = np.nonzero(adjacency_matrix[node.id])[0].tolist()
        self._topology = adjacency_matrix
    
    def train(self, epochs: int) -> List[EpochMetrics]:
        """Run the full training loop."""
        self._notify_callbacks("on_train_start")
        
        for epoch in range(epochs):
            metrics = self._train_epoch(epoch)
            self._metrics_history.append(metrics)
            self._notify_callbacks("on_epoch_end", metrics)
        
        self._notify_callbacks("on_train_end")
        return self._metrics_history
    
    def _train_epoch(self, epoch: int) -> EpochMetrics:
        """Execute one training epoch."""
        # Local training phase
        losses = self._run_local_training()
        
        # Consensus phase
        if (epoch + 1) % self.config.consensus_interval == 0:
            consensus_time = self._run_consensus()
        else:
            consensus_time = 0.0
        
        return EpochMetrics(
            epoch=epoch,
            losses=losses,
            consensus_time=consensus_time,
        )
    
    def _run_local_training(self) -> np.ndarray:
        """Train all nodes locally for one epoch."""
        losses = np.zeros(len(self.nodes))
        
        # Group by device for efficient parallel execution
        for batch in self._batch_by_device():
            self._train_batch_parallel(batch, losses)
        
        return losses
    
    def _run_consensus(self) -> float:
        """Execute the consensus protocol."""
        import time
        start = time.time()
        
        # Reset buffers
        for node in self.nodes:
            node.reset_buffer()
        
        # Prepare and exchange broadcasts
        broadcasts = {
            node.id: node.prepare_broadcast() 
            for node in self.nodes
        }
        
        for node in self.nodes:
            for neighbor_id in node.neighbors:
                node.receive(broadcasts[neighbor_id])
            node.receive(broadcasts[node.id])  # Include self
        
        # Aggregate and update
        for node in self.nodes:
            new_weights = node.aggregate(self.aggregator)
            node.update_weights(new_weights)
        
        return time.time() - start
    
    def _notify_callbacks(self, event: str, *args) -> None:
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(*args)