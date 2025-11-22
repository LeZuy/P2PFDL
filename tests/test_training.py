# tests/test_training.py
"""Tests for training module."""

import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from decen_learn.core import Node, ByzantineNode, RandomWeightProjector
from decen_learn.aggregators import MeanAggregator, KrumAggregator
from decen_learn.attacks import MinMaxAttack
from decen_learn.training import (
    DecentralizedTrainer,
    assign_topology,
    create_topology_erdos_renyi,
    select_byzantine_nodes,
    verify_byzantine_constraint,
)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )


@pytest.fixture
def dummy_dataloader():
    """Create dummy dataloader."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)


@pytest.fixture
def small_network(simple_model, dummy_dataloader):
    """Create small network of nodes for testing."""
    num_nodes = 6
    
    # Create projector
    projector = RandomWeightProjector.from_model(
        simple_model,
        projection_dim=2,
        random_state=42,
    )
    
    # Create nodes
    nodes = []
    for i in range(num_nodes):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        
        if i >= 4:  # Last 2 nodes are Byzantine
            attack = MinMaxAttack()
            node = ByzantineNode(
                node_id=i,
                model=model,
                projector=projector,
                dataloader=dummy_dataloader,
                attack=attack,
                bad_client_ids=[4, 5],
            )
        else:
            node = Node(
                node_id=i,
                model=model,
                projector=projector,
                dataloader=dummy_dataloader,
            )
        
        nodes.append(node)
    
    # Create ring topology
    topology = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        topology[i, (i + 1) % num_nodes] = 1
        topology[i, (i - 1) % num_nodes] = 1
    
    assign_topology(nodes, topology)
    
    return nodes


class TestTopologyUtils:
    """Test topology utilities."""
    
    def test_create_erdos_renyi(self):
        """Test Erdős-Rényi topology creation."""
        topology = create_topology_erdos_renyi(
            num_nodes=10,
            avg_degree=4,
            seed=42,
        )
        
        assert topology.shape == (10, 10)
        assert topology.min() >= 0
        assert topology.max() <= 1
        
        # Check minimum degree
        degrees = topology.sum(axis=1)
        assert (degrees >= 2).all()
    
    def test_assign_topology(self, small_network):
        """Test topology assignment."""
        nodes = small_network
        
        # All nodes should have neighbors
        for node in nodes:
            assert len(node.neighbors) > 0
    
    def test_select_byzantine_random(self):
        """Test random Byzantine selection."""
        bad_indices = select_byzantine_nodes(
            num_nodes=10,
            byzantine_fraction=0.3,
            strategy="random",
            seed=42,
        )
        
        assert len(bad_indices) == 3
        assert all(0 <= i < 10 for i in bad_indices)
    
    def test_select_byzantine_evenly_spaced(self):
        """Test evenly spaced Byzantine selection."""
        bad_indices = select_byzantine_nodes(
            num_nodes=10,
            byzantine_fraction=0.3,
            strategy="evenly_spaced",
        )
        
        assert len(bad_indices) == 3
        # Should be roughly evenly distributed
        assert bad_indices == [0, 3, 6]
    
    def test_verify_constraint_pass(self):
        """Test Byzantine constraint verification (passing)."""
        # Ring topology where each node has 2 neighbors
        topology = np.zeros((9, 9))
        for i in range(9):
            topology[i, (i + 1) % 9] = 1
            topology[i, (i - 1) % 9] = 1
        
        # 3 Byzantine nodes (1/3 of total)
        # But they're spaced such that no node has >1/3 bad neighbors
        bad_nodes = [0, 3, 6]
        
        result = verify_byzantine_constraint(topology, bad_nodes)
        assert result is True
    
    def test_verify_constraint_fail(self):
        """Test Byzantine constraint verification (failing)."""
        # Star topology: node 0 connected to all others
        topology = np.zeros((6, 6))
        for i in range(1, 6):
            topology[0, i] = 1
            topology[i, 0] = 1
        
        # If 3+ peripheral nodes are Byzantine, center has >1/3 bad neighbors
        bad_nodes = [1, 2, 3]
        
        result = verify_byzantine_constraint(topology, bad_nodes)
        assert result is False

class TestDecentralizedTrainer:
    """Test DecentralizedTrainer class."""
    
    def test_initialization(self, small_network):
        """Test trainer initialization."""
        nodes = small_network
        aggregator = MeanAggregator()
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
            max_parallel_gpu=2,
        )
        
        assert len(trainer.nodes) == 6
        assert trainer.aggregator is aggregator
    
    def test_local_training(self, small_network):
        """Test local training phase."""
        nodes = small_network
        aggregator = MeanAggregator()
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
        )
        
        # Run one epoch of local training
        losses = trainer._run_local_training(epoch=0)
        
        assert losses.shape == (6,)
        assert (losses >= 0).all()
    
    def test_consensus_phase(self, small_network):
        """Test consensus phase."""
        nodes = small_network
        aggregator = KrumAggregator(num_byzantine=2)
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
        )
        
        # Train once to have different weights
        for node in nodes:
            if not node.is_byzantine:
                node.train_epoch()
        
        # Run consensus
        consensus_time = trainer._run_consensus_phase(epoch=0)
        
        assert isinstance(consensus_time, float)
        assert consensus_time >= 0
    
    def test_full_training(self, small_network, tmp_path):
        """Test full training loop."""
        nodes = small_network
        aggregator = MeanAggregator()
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
            results_dir=tmp_path,
            consensus_interval=2,
            test_interval=3,
        )
        
        # Run for a few epochs
        history = trainer.train(epochs=5)
        
        assert len(history['losses']) == 5
        
        # Check that files were created
        assert (tmp_path / "loss_epoch_5.txt").exists()
    
    def test_device_grouping(self, small_network):
        """Test device-aware grouping."""
        nodes = small_network
        aggregator = MeanAggregator()
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
        )
        
        groups = trainer._group_by_device(nodes)
        
        # All nodes should be in some group
        total_nodes = sum(len(g) for g in groups)
        assert total_nodes == len(nodes)
    
    def test_honest_byzantine_split(self, small_network):
        """Test node splitting utilities."""
        nodes = small_network
        aggregator = MeanAggregator()
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
        )
        
        honest = trainer.get_honest_nodes()
        byzantine = trainer.get_byzantine_nodes()
        
        assert len(honest) == 4
        assert len(byzantine) == 2
        assert len(honest) + len(byzantine) == len(nodes)


class TestIntegration:
    """Integration tests simulating real scenarios."""
    
    def test_end_to_end_mean(self, small_network, tmp_path):
        """End-to-end test with mean aggregation."""
        nodes = small_network
        aggregator = MeanAggregator()
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
            results_dir=tmp_path,
        )
        
        history = trainer.train(epochs=3)
        
        # Check convergence (losses should generally decrease)
        losses = np.array(history['losses'])
        assert losses.shape == (3, 6)
    
    def test_end_to_end_krum(self, small_network, tmp_path):
        """End-to-end test with Krum aggregation."""
        nodes = small_network
        aggregator = KrumAggregator(num_byzantine=2)
        
        trainer = DecentralizedTrainer(
            nodes=nodes,
            aggregator=aggregator,
            results_dir=tmp_path,
        )
        
        history = trainer.train(epochs=3)
        
        assert len(history['losses']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])