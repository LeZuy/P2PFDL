# main.py (refactored)
"""Entry point using refactored training module."""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import torch

from decen_learn.core import Node, ByzantineNode, RandomWeightProjector
from decen_learn.aggregators import get_aggregator
from decen_learn.attacks import LIEAttack, IPMAttack, MinMaxAttack
from decen_learn.config import ExperimentConfig
from decen_learn.training import (
    DecentralizedTrainer,
    assign_topology,
    load_topology,
    create_topology_ring,
    create_topology_erdos_renyi,
    select_byzantine_nodes,
    verify_byzantine_constraint,
    print_training_summary,
)
from decen_learn.models import ResNet18_CIFAR
from decen_learn.data import get_trainloader, get_testloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decentralized Byzantine-resilient training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--consensus",
        type=str,
        choices=["mean", "krum", "tverberg", "trimmed_mean"],
        help="Override consensus aggregation method"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Maximum number of GPUs to use in parallel"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        help="Number of nodes in network"
    )
    parser.add_argument(
        "--byzantine-fraction",
        type=float,
        help="Fraction of Byzantine nodes"
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=["erdos_renyi", "ring", "file"],
        help="Network topology type"
    )
    parser.add_argument(
        "--topology-file",
        type=str,
        help="Path to topology file (if topology=file)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory with partitioned data"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory for saving results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--projection-file",
        type=str,
        help="Path to saved projection matrix (.npz). If omitted, uses configs/projection_mats.npz when available."
    )
    return parser.parse_args()


def load_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Load experiment configuration and apply CLI overrides."""
    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config))
    else:
        config = ExperimentConfig()
    
    if args.consensus:
        config.consensus_type = args.consensus
    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.num_nodes is not None:
        config.topology.num_nodes = args.num_nodes
    if args.byzantine_fraction is not None:
        config.attack.byzantine_fraction = args.byzantine_fraction
    if args.topology:
        config.topology.type = args.topology
    if args.topology_file:
        config.topology_file = Path(args.topology_file)
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.results_dir:
        config.results_dir = Path(args.results_dir)
    if args.seed is not None:
        config.seed = args.seed
        config.topology.seed = args.seed
    if args.projection_file:
        config.projection_path = Path(args.projection_file)
    
    config._normalize_paths()
    return config


def build_attack(config: ExperimentConfig, num_byzantine: int, num_total: int):
    """Instantiate attack strategy based on config."""
    attack_type = (config.attack.attack_type or "minmax").lower()
    if attack_type == "minmax":
        return MinMaxAttack(
            boosting_factor=1.0,
            gamma_init=20.0,
            tau=1e-3,
        )
    if attack_type == "ipm":
        return IPMAttack()
    if attack_type == "lie":
        return LIEAttack(
            num_byzantine=num_byzantine,
            num_total=num_total,
        )
    raise ValueError(
        f"Unsupported attack_type '{config.attack.attack_type}'. "
        "Choose from ['minmax', 'ipm', 'lie']."
    )

def load_bad_client_ids(config: ExperimentConfig) -> Optional[list]:
    """Return explicit bad client ids from config if provided.
    
    Supports a list in YAML or a path string to a newline-separated file.
    """
    bad_clients = config.attack.bad_clients
    if not bad_clients:
        return None
    
    if isinstance(bad_clients, str):
        path = Path(bad_clients)
        if not path.exists():
            raise ValueError(f"Bad clients file not found: {path}")
        with open(path) as f:
            ids = [int(line.strip()) for line in f if line.strip()]
    elif isinstance(bad_clients, (list, tuple)):
        try:
            ids = [int(x) for x in bad_clients]
        except ValueError:
            raise ValueError("bad_clients must be integers or paths to a file.")
    else:
        raise ValueError("bad_clients must be a list/tuple or filepath string.")
    
    # Deduplicate and keep order
    seen = set()
    uniq_ids = []
    for i in ids:
        if i not in seen:
            uniq_ids.append(i)
            seen.add(i)
    return uniq_ids

def create_nodes(
    num_nodes: int,
    bad_indices: list,
    projector: RandomWeightProjector,
    data_dir: Path,
    config: ExperimentConfig,
    attack,
) -> list:
    """Create network of honest and Byzantine nodes.
    
    Args:
        num_nodes: Total number of nodes
        bad_indices: Indices of Byzantine nodes
        projector: Shared weight projector
        data_dir: Data directory
        config: Experiment configuration
        attack: Instantiated attack strategy
        
    Returns:
        List of nodes
    """
    print(f"\nCreating {num_nodes} nodes...")
    print(f"  Byzantine nodes: {len(bad_indices)}")
    
    nodes = []
    for node_id in range(num_nodes):
        # Create model
        model = ResNet18_CIFAR()
        
        # Create dataloader
        dataloader = get_trainloader(
            f"{data_dir}/client_{node_id}",
            batch_size=config.training.batch_size,
        )
        
        # Create node (Byzantine or honest)
        if node_id in bad_indices:
            node = ByzantineNode(
                node_id=node_id,
                model=model,
                projector=projector,
                dataloader=dataloader,
                attack=attack,
                bad_client_ids=bad_indices,
                learning_rate=config.training.learning_rate,
                momentum=config.training.momentum,
                weight_decay=config.training.weight_decay,
            )
        else:
            node = Node(
                node_id=node_id,
                model=model,
                projector=projector,
                dataloader=dataloader,
                learning_rate=config.training.learning_rate,
                momentum=config.training.momentum,
                weight_decay=config.training.weight_decay,
            )
        
        nodes.append(node)
    
    print(f"✓ Created {len(nodes)} nodes")
    return nodes


def create_topology_matrix(
    num_nodes: int,
    topology_type: str,
    topology_file: Optional[str] = None,
    avg_degree: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Create or load network topology.
    
    Args:
        num_nodes: Number of nodes
        topology_type: Type of topology
        topology_file: Path to file (if type=file)
        avg_degree: Average degree for random topologies
        seed: Random seed
        
    Returns:
        Adjacency matrix
    """
    print(f"\nCreating topology: {topology_type}")
    
    if topology_type == "file":
        if topology_file is None:
            raise ValueError("topology_file required for topology=file")
        topology = load_topology(Path(topology_file))
    elif topology_type == "erdos_renyi":
        topology = create_topology_erdos_renyi(
            num_nodes=num_nodes,
            avg_degree=avg_degree,
            seed=seed,
        )
    elif topology_type == "ring":
        topology = create_topology_ring(
            num_nodes=num_nodes,
            degree=avg_degree,
        )
    else:
        raise ValueError(f"Unknown topology: {topology_type}")
    
    return topology


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config_from_args(args)
    
    # Set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Load or create config
    default_projection_path = Path("configs/projection_mats.npz")
    
    # Handle projection path overrides and defaults
    if config.projection_path is None and default_projection_path.exists():
        config.projection_path = default_projection_path
    
    if config.topology.type == "file" and config.topology_file is None:
        raise ValueError("topology_file must be provided when topology.type is 'file'")
    
    # Create projector
    print("\nInitializing weight projector...")
    base_model = ResNet18_CIFAR()
    base_param_count = base_model.get_num_parameters()
    projector = None
    
    # Try loading precomputed projection matrix
    if config.projection_path:
        proj_path = Path(config.projection_path)
        if proj_path.exists():
            projector = RandomWeightProjector.load(proj_path)
            if projector.original_dim != base_param_count:
                raise ValueError(
                    f"Projection matrix dimension mismatch: expected "
                    f"{base_param_count}, got {projector.original_dim}"
                )
            print(f"✓ Loaded projector from {proj_path}: "
                  f"{base_param_count:,} params → {projector.projection_dim}D")
        else:
            print(f"⚠️ Projection file not found at {proj_path}, "
                  "creating new random projector.")
    
    # Fallback: create new random projector
    if projector is None:
        projector = RandomWeightProjector.from_model(
            base_model,
            projection_dim=config.projection_dim,
            random_state=config.seed,
        )
        save_path = config.projection_path or default_projection_path
        projector.save(save_path)
        config.projection_path = save_path
        print(f"✓ Projector created: {base_param_count:,} params → "
              f"{config.projection_dim}D")
    
    # Select Byzantine nodes (use explicit list if provided)
    bad_indices = load_bad_client_ids(config)
    if bad_indices is None:
        bad_indices = select_byzantine_nodes(
            num_nodes=config.topology.num_nodes,
            byzantine_fraction=config.attack.byzantine_fraction,
            strategy="random",
            seed=config.seed,
        )
    else:
        # Validate provided IDs
        out_of_range = [
            i for i in bad_indices
            if i < 0 or i >= config.topology.num_nodes
        ]
        if out_of_range:
            raise ValueError(
                f"bad_clients contains ids outside [0, {config.topology.num_nodes - 1}]: "
                f"{out_of_range}"
            )
    
    topology_seed = config.topology.seed if config.topology.seed is not None else config.seed
    # Create topology
    topology = create_topology_matrix(
        num_nodes=config.topology.num_nodes,
        topology_type=config.topology.type,
        topology_file=config.topology_file,
        avg_degree=config.topology.degree,
        seed=topology_seed,
    )
    
    # Create attack strategy
    attack = build_attack(
        config=config,
        num_byzantine=len(bad_indices),
        num_total=config.topology.num_nodes,
    )
    
    # Verify Byzantine constraint
    verify_byzantine_constraint(
        adjacency_matrix=topology,
        bad_nodes=bad_indices,
        max_bad_neighbors_fraction=1/3,
    )
    
    # Create nodes
    nodes = create_nodes(
        num_nodes=config.topology.num_nodes,
        bad_indices=bad_indices,
        projector=projector,
        data_dir=config.data_dir,
        config=config,
        attack=attack,
    )
    
    # Assign topology
    assign_topology(nodes, topology)
    
    # Create aggregator
    num_byzantine = len(bad_indices)
    aggregator = get_aggregator(
        config.consensus_type,
        num_byzantine=num_byzantine,
    )
    
    # Create test loader
    test_loader = get_testloader(
        f"{config.data_dir}/test",
        batch_size=config.training.eval_batch_size,
    )
    
    # Print summary
    print_training_summary(nodes, config)
    
    # Create trainer
    trainer = DecentralizedTrainer(
        nodes=nodes,
        aggregator=aggregator,
        test_loader=test_loader,
        results_dir=config.results_dir / config.consensus_type,
        max_parallel_gpu=config.num_gpus,
        consensus_interval=config.training.consensus_interval,
        test_interval=config.training.test_interval,
    )
    
    # Run training
    history = trainer.train(epochs=config.training.epochs)
    
    # Save final models
    trainer.save_final_models()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Results saved to: {config.results_dir / config.consensus_type}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
