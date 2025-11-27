"""Tests for the lightweight ResNet variant used on CIFAR-sized inputs."""

import torch
import pytest

from decen_learn.models.resnet_cifar import BasicBlock, ResNet, resnet20


@pytest.fixture
def cifar_batch():
    """Return a random CIFAR-like batch."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def resnet_model():
    """Return a freshly initialized ResNet20 instance."""
    model = resnet20()
    assert isinstance(model, ResNet)
    return model


def test_resnet20_forward_shape(resnet_model, cifar_batch):
    """The forward pass should keep batch size and map to 10 classes."""
    resnet_model.eval()
    with torch.no_grad():
        logits = resnet_model(cifar_batch)
    assert logits.shape == (cifar_batch.size(0), 10)
    assert torch.isfinite(logits).all()


def test_resnet_parameter_count_matches_manual_sum(resnet_model):
    """get_num_parameters should match a manual parameter count."""
    manual = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    assert resnet_model.get_num_parameters() == manual
    assert manual > 0


def test_basicblock_downsamples_with_option_a():
    """Downsampling shortcut (option A) should resize features correctly."""
    block = BasicBlock(in_planes=16, planes=32, stride=2, option='A')
    x = torch.randn(2, 16, 32, 32)
    y = block(x)
    assert y.shape == (2, 32, 16, 16)
    assert torch.isfinite(y).all()


def test_resnet_repr_includes_parameter_count(resnet_model):
    """__repr__ should expose the formatted parameter count."""
    representation = repr(resnet_model)
    assert "ResNet_CIFAR" in representation
    assert "num_params=" in representation
