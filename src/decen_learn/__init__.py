# src/decen_learn/__init__.py
"""Decentralized Byzantine-resilient federated learning."""

__version__ = "0.1.0"

from .aggregators import get_aggregator, AGGREGATORS
from .config import ExperimentConfig

__all__ = [
    "get_aggregator",
    "AGGREGATORS",
    "ExperimentConfig",
]