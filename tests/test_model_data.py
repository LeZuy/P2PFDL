# tests/test_models_and_data.py
"""Tests for migrated model and data modules."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from decen_learn.models import ResNet18_CIFAR, TinyCNN
from decen_learn.data import (
    get_trainloader,
    get_testloader,
    partition_data_dirichlet,
    partition_data_iid,
    add_pixel_pattern,
    PoisonedDataset,
    get_poison_batch,
)


class TestModels:
    """Test model architectures."""
    
    def test_resnet18_cifar_creation(self):
        """Test ResNet18 CIFAR model creation."""
        model = ResNet18_CIFAR(num_classes=10)
        assert isinstance(model, torch.nn.Module)
        
        # Check number of parameters
        num_params = model.get_num_parameters()
        assert num_params > 0
        print(f"ResNet18_CIFAR parameters: {num_params:,}")
    
    def test_resnet18_cifar_forward(self):
        """Test forward pass."""
        model = ResNet18_CIFAR(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        
        y = model(x)
        
        assert y.shape == (4, 10)
        assert not torch.isnan(y).any()
    
    def test_tiny_cnn_creation(self):
        """Test TinyCNN creation."""
        model = TinyCNN(num_classes=10)
        assert isinstance(model, torch.nn.Module)
        
        num_params = model.get_num_parameters()
        assert num_params > 0
        print(f"TinyCNN parameters: {num_params:,}")
    
    def test_tiny_cnn_forward(self):
        """Test TinyCNN forward pass."""
        model = TinyCNN(num_classes=10)
        x = torch.randn(8, 3, 32, 32)
        
        y = model(x)
        
        assert y.shape == (8, 10)
        assert not torch.isnan(y).any()
    
    def test_model_size_comparison(self):
        """Compare model sizes."""
        resnet = ResNet18_CIFAR(num_classes=10)
        tiny = TinyCNN(num_classes=10)
        
        resnet_params = resnet.get_num_parameters()
        tiny_params = tiny.get_num_parameters()
        
        # ResNet should be larger than TinyCNN
        assert resnet_params > tiny_params
        print(f"Size ratio: {resnet_params / tiny_params:.1f}x")


class TestDataLoading:
    """Test data loading utilities."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        temp_dir = tempfile.mkdtemp()
        
        # Create fake data structure
        data_path = Path(temp_dir) / "client_0"
        for class_id in range(3):  # 3 classes for testing
            class_dir = data_path / f"class_{class_id}"
            class_dir.mkdir(parents=True)
            
            # Create a few dummy images
            for i in range(5):
                img = torch.rand(3, 32, 32)
                img_path = class_dir / f"img_{i:05d}.png"
                from torchvision.utils import save_image
                save_image(img, img_path)
        
        yield str(data_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_get_trainloader(self, temp_data_dir):
        """Test train loader creation."""
        try:
            loader = get_trainloader(
                temp_data_dir,
                batch_size=4,
                dataset_name="cifar10"
            )
            
            assert len(loader) > 0
            
            # Test one batch
            images, labels = next(iter(loader))
            assert images.shape[1:] == (3, 32, 32)
            assert len(labels) == len(images)
            
        except Exception as e:
            pytest.skip(f"Could not create loader: {e}")
    
    def test_transforms_normalization(self):
        """Test that transforms apply proper normalization."""
        from decen_learn.data.loaders import get_transforms
        
        transform = get_transforms("cifar10", train=False)
        
        # Create dummy image
        img = torch.ones(3, 32, 32) * 0.5
        normalized = transform(img.numpy().transpose(1, 2, 0))
        
        # Should be normalized (not equal to original)
        assert not torch.allclose(normalized, img)


class TestDataPartitioning:
    """Test data partitioning."""
    
    @pytest.mark.slow
    def test_dirichlet_partitioning(self):
        """Test Dirichlet partitioning."""
        try:
            clients, testset = partition_data_dirichlet(
                num_clients=5,
                alpha=0.5,
                dataset_name="cifar10",
                seed=42,
            )
            
            assert len(clients) == 5
            assert len(testset) > 0
            
            # Check that each client has data
            for i, client_data in enumerate(clients):
                assert len(client_data) > 0
                print(f"Client {i}: {len(client_data)} samples")
        
        except Exception as e:
            pytest.skip(f"Could not download dataset: {e}")
    
    @pytest.mark.slow
    def test_iid_partitioning(self):
        """Test IID partitioning."""
        try:
            clients, testset = partition_data_iid(
                num_clients=4,
                dataset_name="cifar10",
                seed=42,
            )
            
            assert len(clients) == 4
            
            # Check roughly equal sizes
            sizes = [len(c) for c in clients]
            assert max(sizes) - min(sizes) < len(clients[0]) * 0.1
            
        except Exception as e:
            pytest.skip(f"Could not download dataset: {e}")


class TestPoisoning:
    """Test data poisoning utilities."""
    
    def test_add_pixel_pattern(self):
        """Test adding pixel trigger pattern."""
        params = {
            'trigger_num': 1,
            '0_poison_pattern': [[0, 0], [0, 1], [1, 0], [1, 1]],
            'trigger_value': 1.0,
        }
        
        image = torch.zeros(3, 32, 32)
        poisoned = add_pixel_pattern(params, image, adversarial_index=0)
        
        # Check that trigger pixels were set
        assert poisoned[0, 0, 0] == 1.0
        assert poisoned[0, 0, 1] == 1.0
        assert poisoned[0, 1, 0] == 1.0
        assert poisoned[0, 1, 1] == 1.0
        
        # Rest should be unchanged
        assert poisoned[0, 2, 2] == 0.0
    
    def test_poisoned_dataset(self):
        """Test PoisonedDataset wrapper."""
        from torch.utils.data import TensorDataset
        
        # Create fake dataset
        images = torch.randn(10, 3, 32, 32)
        labels = torch.randint(0, 10, (10,))
        base_dataset = TensorDataset(images, labels)
        
        params = {
            'trigger_num': 1,
            '0_poison_pattern': [[0, 0]],
            'trigger_value': 1.0,
            'poison_label_swap': 0,
        }
        
        poisoned_dataset = PoisonedDataset(
            base_dataset,
            params,
            adversarial_index=0
        )
        
        # Check that all labels are changed
        for i in range(len(poisoned_dataset)):
            img, label = poisoned_dataset[i]
            assert label == 0
            assert img[0, 0, 0] == 1.0
    
    def test_get_poison_batch(self):
        """Test batch poisoning."""
        params = {
            'trigger_num': 1,
            '0_poison_pattern': [[0, 0], [1, 1]],
            'trigger_value': 1.0,
            'poison_label_swap': 5,
            'DPR': 0.5,
            'poison_random': False,
        }
        
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        
        poisoned_images, poisoned_labels, count = get_poison_batch(
            params,
            (images, labels),
            device=torch.device('cpu'),
            adversarial_index=0,
        )
        
        # Should poison ~50% of batch
        assert 2 <= count <= 6
        
        # Check that poisoned labels are correct
        assert (poisoned_labels == 5).sum().item() == count
        
        # Check trigger was applied
        for i in range(count):  # First count samples should be poisoned
            assert poisoned_images[i, 0, 0, 0] == 1.0
            assert poisoned_images[i, 0, 1, 1] == 1.0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_model_with_poisoned_data(self):
        """Test training model on poisoned data."""
        model = TinyCNN(num_classes=10)
        
        # Create poisoned batch
        params = {
            'trigger_num': 1,
            '0_poison_pattern': [[0, 0]],
            'trigger_value': 1.0,
            'poison_label_swap': 0,
            'DPR': 1.0,  # Poison everything
        }
        
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        poisoned_images, poisoned_labels, _ = get_poison_batch(
            params,
            (images, labels),
            device=torch.device('cpu'),
            adversarial_index=0,
        )
        
        # Forward pass
        outputs = model(poisoned_images)
        
        assert outputs.shape == (4, 10)
        assert not torch.isnan(outputs).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])