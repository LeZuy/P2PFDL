# src/decen_learn/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml

@dataclass
class TopologyConfig:
    type: str = "erdos_renyi"
    num_nodes: int = 64
    degree: int = 6
    seed: int = 42

@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    consensus_interval: int = 1
    test_interval: int = 5

@dataclass
class AttackConfig:
    byzantine_fraction: float = 0.33
    bad_clients: Optional[List[int]] = None
    attack_type: str = "minmax"

@dataclass
class ExperimentConfig:
    name: str = "default"
    consensus_type: str = "mean"
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    data_dir: Path = Path("./data_splits")
    results_dir: Path = Path("./results")
    projection_dim: int = 2
    num_gpus: int = 4
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)