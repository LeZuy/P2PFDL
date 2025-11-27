# src/decen_learn/training/trainer.py
"""Main trainer orchestrating decentralized training with consensus."""

from __future__ import annotations

import time
import copy
from pathlib import Path
from typing import List, Optional, Sequence, Callable
from collections import deque, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core import Node
from ..aggregators import BaseAggregator

class DecentralizedTrainer:
    """Orchestrates decentralized training with periodic consensus.
    
    Key responsibilities:
    - Coordinate local training across nodes
    - Execute consensus rounds
    - Manage evaluation
    - Handle callbacks and logging
    - Optimize parallel execution (Serialized for stability on Single GPU/ROCm)
    """
    
    def __init__(
        self,
        nodes: List[Node],
        aggregator: BaseAggregator,
        test_loader: Optional[DataLoader] = None,
        results_dir: Optional[Path] = None,
        max_parallel_gpu: int = 4,
        consensus_interval: int = 1,
        test_interval: int = 5,
        release_interval: int = 1,
        projection_snapshot_interval: int = 20,
        early_projection_epochs: int = 10,
        enable_global_attack: bool = False,
    ):
        """Initialize trainer.
        
        Args:
            nodes: List of participating nodes
            aggregator: Aggregation strategy
            test_loader: Test dataloader for evaluation
            results_dir: Directory for saving results
            max_parallel_gpu: Max GPU nodes to process in a batch (serialized execution)
            consensus_interval: Run consensus every N epochs
            test_interval: Run evaluation every N epochs
            release_interval: Move tracked models back to CPU every N epochs
            projection_snapshot_interval: Save projections every N epochs
            early_projection_epochs: Save projections for first N epochs
            enable_global_attack: Allow Byzantine nodes to view all honest weights
                when crafting malicious updates
        """
        self.nodes = nodes
        self.aggregator = aggregator
        self.test_loader = test_loader
        self.results_dir = Path(results_dir) if results_dir else Path("./results")
        self.max_parallel_gpu = max_parallel_gpu
        self.consensus_interval = consensus_interval
        self.test_interval = test_interval
        self.release_interval = max(1, release_interval)
        self.projection_snapshot_interval = projection_snapshot_interval
        self.early_projection_epochs = early_projection_epochs
        self.enable_global_attack = enable_global_attack
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if Tverberg (needs special handling for preimage)
        self.is_tverberg = "tverberg" in aggregator.__class__.__name__.lower()
        
        # Metrics tracking
        self.losses_history = []
        self.test_metrics_history = []
        self.test_epochs: List[int] = []
        
        print(f"Initialized trainer with {len(nodes)} nodes")
        print(f"\tAggregator: {aggregator.__class__.__name__}")
        print(f"\tMax batched nodes: {max_parallel_gpu}")
        print(f"\tResults dir: {self.results_dir}")
    
    def train(self, epochs: int) -> dict:
        """Run full training loop.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training history and metrics
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # ===== Local Training Phase =====
            losses = self._run_local_training(epoch)
            self.losses_history.append(losses)
            
            # ===== Consensus Phase =====
            if (epoch + 1) % self.consensus_interval == 0:
                consensus_time = self._run_consensus_phase(epoch)
            else:
                consensus_time = 0.0
            if epoch % 5 == 0:
                print(f"Consensus time: {consensus_time:.2f}s")
            
            # ===== Evaluation Phase =====
            if epoch % self.test_interval == 0:
                self.test_epochs.append(epoch)
                test_metrics = self._run_evaluation(epoch)
                self.test_metrics_history.append(test_metrics)
            
            if epoch % self.release_interval == 0:
                for node in self.nodes:
                    node.device_manager.release_all()

            epoch_time = time.time() - epoch_start
            
            # Print summary
            # if epoch % 5 == 0:
            #     avg_loss = np.mean(losses)
            #     print(
            #         f"Epoch {epoch+1:3d}/{epochs} | "
            #         f"Loss: {avg_loss:.4f} | "
            #         f"Time: {epoch_time:.2f}s"
            #     )
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}\n")
        self._save_history(epochs)

        return {
            'losses': self.losses_history,
            'test_metrics': self.test_metrics_history,
        }

    def _save_history(self, epochs: int) -> None:
        """Persist aggregate training and testing metrics for each epoch."""
        if not self.losses_history:
            return

        losses_array = np.array(self.losses_history)
        avg_train_losses = losses_array.mean(axis=1)

        # Initialize with NaNs for epochs without evaluation
        test_loss_means = np.full(epochs, np.nan)
        test_acc_means = np.full(epochs, np.nan)

        for epoch, metrics in zip(self.test_epochs, self.test_metrics_history):
            # metrics shape: (num_nodes, 3) -> [loss, acc, asr]
            mean_metrics = metrics.mean(axis=0)
            test_loss_means[epoch] = mean_metrics[0]
            test_acc_means[epoch] = mean_metrics[1]

        history = np.column_stack(
            (
                np.arange(1, epochs + 1),
                avg_train_losses,
                test_loss_means,
                test_acc_means,
            )
        )

        np.savetxt(
            self.results_dir / "training_history.csv",
            history,
            delimiter=",",
            header="epoch,train_loss,test_loss,test_accuracy",
            comments="",
            fmt=["%d", "%.6f", "%.6f", "%.6f"],
        )
    
    def _run_local_training(self, epoch: int) -> np.ndarray:
        """Train all nodes locally for one epoch.
        
        IMPORTANT: Executes sequentially to avoid ROCm/GPU context crashes on RDNA3.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Array of training losses for each node
        """
        losses = np.zeros(len(self.nodes))
        
       # Group nodes by device
        device_groups = self._group_by_device(self.nodes)
        
        for batch in self._round_robin_batches(device_groups):
            # Process nodes in the batch sequentially
            for node in batch:
                try:
                    loss, _ = node.train_epoch()
                    losses[node.id] = loss
                except Exception as e:
                    print(f"[Epoch {epoch}] Error training node {node.id}: {e}")
                    raise e
        
        # Save losses
        # np.savetxt(
        #     self.results_dir / f"loss_epoch_{epoch + 1}.txt",
        #     losses,
        #     fmt="%.4f"
        # )
        
        return losses
    
    def _run_consensus_phase(self, epoch: int) -> float:
        """Execute consensus protocol.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Time taken for consensus (seconds)
        """
        print(f"\n=== Consensus phase after Epoch {epoch + 1} ===")
        start_time = time.time()
        
        # Step 1: Reset buffers
        for node in self.nodes:
            node.reset_buffers()
            if node.is_byzantine and hasattr(node, "set_global_honest_cache"):
                node.set_global_honest_cache(None)
        
        # Step 2: Prepare broadcasts
        if self.enable_global_attack:
            global_honest = [
                Node._clone_weight_dict(n.state.weights)
                for n in self.nodes
                if not n.is_byzantine
            ]
        else:
            global_honest = None
        if global_honest is not None:
            for n in self.nodes:
                if (
                    n.is_byzantine
                    and hasattr(n, "set_global_honest_cache")
                    and hasattr(n, "_attack")
                    and hasattr(n._attack, "craft_global")
                ):
                    n.set_global_honest_cache(global_honest)

        broadcasts = {}
        for node in self.nodes:
            if node.is_byzantine:
                # Byzantine nodes need access to all nodes
                broadcasts[node.id] = node.prepare_broadcast(
                    processes=self.nodes
                )
            else:
                broadcasts[node.id] = node.prepare_broadcast()
        
        # Step 3: Exchange messages
        for node in self.nodes:
            # Receive from neighbors
            for neighbor_id in node.neighbors:
                sender = self.nodes[neighbor_id]
                # Byzantine nodes don't send to each other
                if node.is_byzantine and sender.is_byzantine:
                    continue
                node.receive(broadcasts[neighbor_id])
            
            # Include self (for fallback in sparse networks)
            node.receive(broadcasts[node.id])
        
        # Step 4: Save pre-consensus projections (if needed)
        record_projection = self._should_snapshot_projection(epoch)
        if record_projection:
            self._save_consensus_payloads(epoch)
            self._save_projected_weights(epoch, "before")
        
        # Step 5: Project buffers (for Tverberg and visualization)
        if self.is_tverberg:
            for node in self.nodes:
                if not node.is_byzantine:
                    node.project_buffers()
        
        # Step 6: Aggregate
        neighbor_counts = [len(node.state.buffer) for node in self.nodes]
        
        consensus_outputs = []
        for node in self.nodes:
            if node.is_byzantine:
                # Byzantine nodes don't aggregate
                consensus_outputs.append(
                    node._clone_weight_dict(node.state.weights)
                )
            else:
                aggregated = node.aggregate(self.aggregator)
                consensus_outputs.append(aggregated)
        
        # Step 7: Save post-consensus projections (if needed)
        if record_projection:
            self._save_projected_weights(epoch, "after")
        
        # Step 8: Update weights
        for node, consensus_output, neighbor_count in zip(
            self.nodes, consensus_outputs, neighbor_counts
        ):
            if node.is_byzantine:
                continue  # Byzantine nodes don't update
            
            # For Tverberg with neighbors, use preimage reconstruction
            if (
                self.is_tverberg
                and neighbor_count > 0
                and node.state.consensus_coefficients is not None
            ):
                # Reconstruct from convex combination
                updated_weights = node._reconstruct_from_coefficients(
                    node.state.consensus_coefficients.get("__flat__")
                )
            else:
                updated_weights = consensus_output
            
            # Update with no blending coefficient (full update)
            node.update_weights(updated_weights, _lambda=0.0)
        
        consensus_time = time.time() - start_time
        print(f"Consensus phase took {consensus_time:.2f}s")
        return consensus_time
    
    def _run_evaluation(self, epoch: int) -> np.ndarray:
        """Evaluate all nodes on test set.
        
        IMPORTANT: Executes sequentially to avoid ROCm/GPU context crashes.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Array of shape (num_nodes, 3) with [loss, acc, asr]
        """
        print(f"\n--- Test results after Epoch {epoch + 1} ---")
        
        if self.test_loader is None:
            print("No test loader provided, skipping evaluation")
            return np.zeros((len(self.nodes), 3))
        
        metrics = np.zeros((len(self.nodes), 3))
        
        # Group by device
        device_groups = self._group_by_device(self.nodes)
        
        for batch in self._round_robin_batches(device_groups):
            # Process nodes in the batch sequentially
            for node in batch:
                loss, acc = node.evaluate(self.test_loader)
                metrics[node.id] = [loss, acc, 0.0]
        
        # Save metrics
        # np.savetxt(
        #     self.results_dir / f"test_results_epoch_{epoch + 1}.txt",
        #     metrics,
        #     fmt="%.4f"
        # )
        
        # Print summary
        avg_acc = np.mean(metrics[:, 1])
        print(f"Average accuracy: {avg_acc:.2f}%")
        
        return metrics
    
    def _save_projected_weights(self, epoch: int, stage: str) -> None:
        """Save projected weights for all nodes.
        
        Args:
            epoch: Current epoch number
            stage: "before" or "after" consensus
        """
        # Determine projection dimension from first node
        if not self.nodes:
            return
        
        first_proj = self.nodes[0].state.projected_weights
        if first_proj is None or first_proj.numel() == 0:
            return
        
        proj_dim = int(first_proj.numel())
        num_nodes = len(self.nodes)
        
        # For now, assume single flattened layer
        # Shape: (num_nodes, 1, proj_dim)
        projected_array = np.zeros((num_nodes, 1, proj_dim), dtype=np.float32)
        
        for i, node in enumerate(self.nodes):
            proj = node.state.projected_weights
            if proj is not None:
                projected_array[i, 0, :] = (
                    proj.detach().cpu().view(-1).numpy()
                )
        
        filename = f"proj_weights_{stage}_epoch_{epoch + 1}.npy"
        np.save(self.results_dir / filename, projected_array)
    
    def _save_consensus_payloads(self, epoch: int) -> None:
        """Save the raw payloads each node received during consensus.
        
        Args:
            epoch: Current epoch number
        """
        payloads = {}
        
        for node in self.nodes:
            if not node.state.buffer_projected:
                continue
            
            # Stack projected payloads
            vectors = torch.stack(
                [
                    proj if torch.is_tensor(proj) else torch.as_tensor(proj)
                    for proj in node.state.buffer_projected
                ],
                dim=0,
            ).detach().cpu().numpy()
            payloads[node.id] = {"__flat__": vectors}
        
        filename = f"payloads_epoch_{epoch + 1}.npy"
        np.save(self.results_dir / filename, payloads, allow_pickle=True)
    
    def _should_snapshot_projection(self, epoch: int) -> bool:
        """Determine if this epoch should save projection snapshots.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            True if snapshots should be saved
        """
        return (
            epoch % self.projection_snapshot_interval == 0
            or epoch < self.early_projection_epochs
        )
    
    def _group_by_device(self, nodes: Sequence[Node]) -> List[deque]:
        """Group nodes by their assigned device.
        
        Args:
            nodes: Sequence of nodes
            
        Returns:
            List of deques, one per device
        """
        buckets = defaultdict(deque)
        for node in nodes:
            device = node.device
            key = (device.type, device.index)
            buckets[key].append(node)
        return list(buckets.values())
    
    def _round_robin_batches(
        self,
        device_groups: List[deque],
    ):
        """Yield batches selecting at most one node per device.
        
        Args:
            device_groups: List of deques of nodes grouped by device
            
        Yields:
            Lists of nodes to process in the current step
        """
        limit = max(1, int(self.max_parallel_gpu))
        
        while True:
            batch = []
            for group in device_groups:
                if not group:
                    continue
                batch.append(group.popleft())
                if len(batch) >= limit:
                    break
            
            if not batch:
                break
            
            yield batch
    
    def save_final_models(self, path: Optional[Path] = None) -> None:
        """Save final model states for all nodes.
        
        Args:
            path: Path to save models (default: results_dir/final_models.pth)
        """
        if path is None:
            path = self.results_dir / "final_models.pth"
        
        state_dicts = {
            f"node_{node.id}": node.model.state_dict()
            for node in self.nodes
        }
        
        torch.save(state_dicts, path)
        print(f"Saved final models to {path}")
    
    def get_honest_nodes(self) -> List[Node]:
        """Return list of honest nodes only."""
        return [node for node in self.nodes if not node.is_byzantine]
    
    def get_byzantine_nodes(self) -> List[Node]:
        """Return list of Byzantine nodes only."""
        return [node for node in self.nodes if node.is_byzantine]
    
    def __repr__(self) -> str:
        num_byzantine = sum(1 for n in self.nodes if n.is_byzantine)
        return (
            f"DecentralizedTrainer("
            f"nodes={len(self.nodes)}, "
            f"byzantine={num_byzantine}, "
            f"aggregator={self.aggregator.__class__.__name__})"
        )


def create_trainer_from_config(
    nodes: List[Node],
    aggregator: BaseAggregator,
    config,  # ExperimentConfig
    test_loader: Optional[DataLoader] = None,
) -> DecentralizedTrainer:
    """Factory function to create trainer from experiment config.
    
    Args:
        nodes: List of nodes
        aggregator: Aggregation strategy
        config: ExperimentConfig instance
        test_loader: Optional test dataloader
        
    Returns:
        Configured trainer
    """
    return DecentralizedTrainer(
        nodes=nodes,
        aggregator=aggregator,
        test_loader=test_loader,
        results_dir=config.results_dir / config.consensus_type,
        max_parallel_gpu=config.num_gpus,
        consensus_interval=config.training.consensus_interval,
        test_interval=config.training.test_interval,
        release_interval=config.training.release_interval,
        enable_global_attack=getattr(config.attack, "global_attack", False),
    )
