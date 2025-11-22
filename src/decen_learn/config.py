# src/decen_learn/config.py
from dataclasses import asdict, dataclass, field
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
    topology_file: Optional[Path] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    seed: int = 42
    data_dir: Path = Path("./data_splits")
    results_dir: Path = Path("./results")
    projection_dim: int = 2
    projection_path: Optional[Path] = None
    num_gpus: int = 4
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}

        topology_data = data.get("topology", {})
        training_data = data.get("training", {})
        attack_data = data.get("attack", {})

        if isinstance(topology_data, dict):
            # Allow users to nest topology_file inside topology, but keep it at root
            topo_file = topology_data.pop("topology_file", None)
            if topo_file and "topology_file" not in data:
                data["topology_file"] = topo_file
            data["topology"] = TopologyConfig(**topology_data)
        if isinstance(training_data, dict):
            data["training"] = TrainingConfig(**training_data)
        if isinstance(attack_data, dict):
            data["attack"] = AttackConfig(**attack_data)

        config = cls(**data)
        config._normalize_paths()
        return config

    def _normalize_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        if self.projection_path:
            self.projection_path = Path(self.projection_path)
        if self.topology_file:
            self.topology_file = Path(self.topology_file)
    
    def save(self, path: Path) -> None:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["results_dir"] = str(self.results_dir)
        data["projection_path"] = (
            str(self.projection_path) if self.projection_path else None
        )
        data["topology_file"] = str(self.topology_file) if self.topology_file else None
        with open(path, "w") as f:
            yaml.safe_dump(data, f)
