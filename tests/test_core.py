# tests/test_core.py
"""Tests for core module."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from decen_learn.core import (
    Node,
    ByzantineNode,
    NodeState,
    RandomWeightProjector,
    IdentityProjector,
    LocalTrainer,
    DeviceManager,
)
from decen_learn.aggregators import MeanAggregator
from decen_learn.attacks import MinMaxAttack


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
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
def projector(simple_model):
    """Create projector."""
    return RandomWeightProjector.from_model(
        simple_model,
        projection_dim=2,
        random_state=42,
    )


class TestNodeState:
    """Test NodeState class."""
    
    def test_initialization(self):
        state = NodeState()
        assert state.loss == 0.0
        assert state.epoch == 0
        assert len(state.buffer) == 0
    
    def test_reset_buffers(self):
        state = NodeState()
        state.buffer = [{"a": torch.tensor([1.0, 2.0, 3.0])}]
        state.buffer_projected = [torch.tensor([1.0, 2.0])]
        
        state.reset_buffers()
        
        assert len(state.buffer) == 0
        assert len(state.buffer_projected) == 0
    
    def test_serialization(self):
        state = NodeState()
        state.weights = {"layer1": torch.tensor([1.0, 2.0, 3.0])}
        state.loss = 0.5
        state.epoch = 10
        
        data = state.to_dict()
        restored = NodeState.from_dict(data)
        
        assert restored.loss == 0.5
        assert restored.epoch == 10
        torch.testing.assert_close(
            restored.weights["layer1"],
            torch.tensor([1.0, 2.0, 3.0])
        )


class TestWeightProjector:
    """Test weight projectors."""
    
    def test_random_projector(self, simple_model):
        projector = RandomWeightProjector.from_model(
            simple_model,
            projection_dim=2,
            random_state=42,
        )
        
        # Create dummy weights
        weights = {
            name: param.detach().clone()
            for name, param in simple_model.named_parameters()
        }
        
        # Project
        projected = projector.project(weights)
        
        assert projected.shape == (2,)
        assert projector.get_projection_dim() == 2
    
    def test_random_projector_layerwise(self, simple_model):
        projector = RandomWeightProjector.from_model(
            simple_model,
            projection_dim=3,
            random_state=7,
        )
        weights = {
            name: param.detach().clone()
            for name, param in simple_model.named_parameters()
        }
        per_layer = projector.project_layerwise(weights)
        assert set(per_layer.keys()) == set(weights.keys())
        for proj in per_layer.values():
            assert proj.shape == (3,)
        combined = torch.stack(list(per_layer.values()), dim=0).sum(dim=0)
        torch.testing.assert_close(combined, projector.project(weights))
    
    def test_identity_projector(self):
        projector = IdentityProjector()
        
        weights = {
            "layer1": np.array([1, 2, 3]),
            "layer2": np.array([4, 5]),
        }
        
        projected = projector.project(weights)
        
        expected = torch.tensor([1, 2, 3, 4, 5])
        torch.testing.assert_close(projected, expected)
    
    def test_identity_projector_layerwise(self):
        projector = IdentityProjector()
        weights = {
            "layer1": torch.tensor([[1.0, 2.0]]),
            "layer2": torch.tensor([3.0, 4.0, 5.0]),
        }
        per_layer = projector.project_layerwise(weights)
        assert set(per_layer.keys()) == set(weights.keys())
        torch.testing.assert_close(
            per_layer["layer1"],
            torch.tensor([1.0, 2.0]),
        )
        torch.testing.assert_close(
            per_layer["layer2"],
            torch.tensor([3.0, 4.0, 5.0]),
        )
    
    def test_projector_save_load(self, simple_model, tmp_path):
        projector = RandomWeightProjector.from_model(
            simple_model,
            projection_dim=3,
            random_state=42,
        )
        
        # Save
        save_path = tmp_path / "projector.npz"
        projector.save(str(save_path))
        
        # Load
        loaded = RandomWeightProjector.load(str(save_path))
        
        assert loaded.projection_dim == 3
        torch.testing.assert_close(
            loaded.projection_matrix,
            projector.projection_matrix,
        )


class TestNode:
    """Test honest Node class."""
    
    def test_initialization(self, simple_model, projector, dummy_dataloader):
        node = Node(
            node_id=0,
            model=simple_model,
            projector=projector,
            dataloader=dummy_dataloader,
        )
        
        assert node.id == 0
        assert not node.is_byzantine
        assert len(node.neighbors) == 0
        assert len(node.state.weights) > 0
    
    def test_train_epoch(self, simple_model, projector, dummy_dataloader):
        node = Node(
            node_id=0,
            model=simple_model,
            projector=projector,
            dataloader=dummy_dataloader,
        )
        
        initial_weights = Node._clone_weight_dict(node.state.weights)
        loss, acc = node.train_epoch()
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert node.state.epoch == 1
        
        # Weights should change after training
        assert not torch.allclose(
            list(initial_weights.values())[0],
            list(node.state.weights.values())[0],
        )
    
    def test_communication(self, simple_model, projector, dummy_dataloader):
        node1 = Node(0, simple_model, projector, dummy_dataloader)
        node2 = Node(1, simple_model, projector, dummy_dataloader)
        
        # Prepare broadcast
        weights = node1.prepare_broadcast()
        
        # Receive
        node2.receive(weights)
        
        assert len(node2.state.buffer) == 1
    
    def test_aggregation(self, simple_model, projector, dummy_dataloader):
        node = Node(0, simple_model, projector, dummy_dataloader)
        
        # Create fake neighbors
        for _ in range(3):
            fake_weights = {
                name: param.detach().clone() + 0.1 * torch.randn_like(param)
                for name, param in simple_model.named_parameters()
            }
            node.receive(fake_weights)
        
        # Aggregate
        aggregator = MeanAggregator()
        aggregated = node.aggregate(aggregator)
        
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0
        for name, tensor in aggregated.items():
            stacked = torch.stack(
                [weights[name] for weights in node.state.buffer],
                dim=0,
            )
            torch.testing.assert_close(
                tensor,
                stacked.mean(dim=0),
                rtol=1e-5,
                atol=1e-6,
            )


class TestByzantineNode:
    """Test Byzantine node class."""
    
    def test_initialization(
        self,
        simple_model,
        projector,
        dummy_dataloader,
    ):
        attack = MinMaxAttack()
        
        node = ByzantineNode(
            node_id=0,
            model=simple_model,
            projector=projector,
            dataloader=dummy_dataloader,
            attack=attack,
            bad_client_ids=[0, 3, 5],
        )
        
        assert node.is_byzantine
        assert node._attack is attack
    
    def test_ignores_updates(
        self,
        simple_model,
        projector,
        dummy_dataloader,
    ):
        attack = MinMaxAttack()
        
        node = ByzantineNode(
            node_id=0,
            model=simple_model,
            projector=projector,
            dataloader=dummy_dataloader,
            attack=attack,
            bad_client_ids=[0],
        )
        
        initial_weights = Node._clone_weight_dict(node.state.weights)
        
        # Try to update (should be ignored)
        fake_weights = {
            name: arr + 1.0
            for name, arr in initial_weights.items()
        }
        node.update_weights(fake_weights)
        
        # Weights should not change
        for name in initial_weights:
            torch.testing.assert_close(
                initial_weights[name],
                node.state.weights[name],
            )


class TestDeviceManager:
    """Test device manager."""
    
    def test_cpu_assignment(self):
        manager = DeviceManager(node_id=0)
        
        # Should default to CPU if no GPU
        device = manager.get_assigned_device()
        assert device.type in ["cpu", "cuda"]
    
    def test_gpu_assignment_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        manager = DeviceManager(node_id=0)
        device = manager.get_assigned_device()
        
        assert device.type == "cuda"
    
    def test_model_movement(self, simple_model):
        manager = DeviceManager(node_id=0)
        
        # Acquire
        device = manager.acquire(simple_model)
        assert manager.get_current_device() == device
        
        # Release
        manager.release(simple_model)
        if device.type == "cuda":
            assert manager.get_current_device().type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
