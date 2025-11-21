# src/decen_learn/training/__init__.py
"""Training orchestration for decentralized learning."""

from .trainer import DecentralizedTrainer, create_trainer_from_config
from .utils import (
    assign_topology,
    load_topology,
    create_topology_erdos_renyi,
    create_topology_ring,
    select_byzantine_nodes,
    verify_byzantine_constraint,
    save_training_metadata,
    load_checkpoint,
    save_checkpoint,
    print_training_summary,
)

__all__ = [
    # Trainer
    "DecentralizedTrainer",
    "create_trainer_from_config",
    # Utilities
    "assign_topology",
    "load_topology",
    "create_topology_erdos_renyi",
    "create_topology_ring",
    "select_byzantine_nodes",
    "verify_byzantine_constraint",
    "save_training_metadata",
    "load_checkpoint",
    "save_checkpoint",
    "print_training_summary",
]