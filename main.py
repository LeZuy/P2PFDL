# main.py (refactored)
"""Entry point using refactored training module."""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import torch

from decen_learn.core import Node, ByzantineNode, RandomWeightProjector
from decen_learn.aggregators import get_aggregator
from decen_learn.attacks import MinMaxAttack
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
from model.ResNet_Cifar import ResNet18_CIFAR
from model.train import get_trainloader, get_testloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decentralized Byzantine-resilient training"
    )
    parser.add_argument(
        "--consensus",
        type=str,
        default="mean",
        choices=["mean", "krum", "tverberg", "trimmed_mean"],
        help="Consensus aggregation method"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Maximum number of GPUs to use in parallel"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=64,
        help="Number of nodes in network"
    )
    parser.add_argument(
        "--byzantine-fraction",
        type=float,
        default=0.33,
        help="Fraction of Byzantine nodes"
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="erdos_renyi",
        choices=["erdos_renyi", "ring", "file"],
        help="Network topology type"
    )
    parser.add_argument(
        "--topology-file",
        type=str,
        default="./configs/erdos_renyi.txt",
        help="Path to topology file (if topology=file)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data_splits",
        help="Directory with partitioned data"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def create_nodes(
    num_nodes: int,
    bad_indices: list,
    projector: RandomWeightProjector,
    data_dir: Path,
    config: ExperimentConfig,
) -> list:
    """Create network of honest and Byzantine nodes.
    
    Args:
        num_nodes: Total number of nodes
        bad_indices: Indices of Byzantine nodes
        projector: Shared weight projector
        data_dir: Data directory
        config: Experiment configuration
        
    Returns:
        List of nodes
    """
    print(f"\nCreating {num_nodes} nodes...")
    print(f"  Byzantine nodes: {len(bad_indices)}")
    
    # Create attack strategy for Byzantine nodes
    attack = MinMaxAttack(
        boosting_factor=1.0,
        gamma_init=20.0,
        tau=1e-3,
    )
    
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
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load or create config
    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config))
    else:
        config = ExperimentConfig()
        config.consensus_type = args.consensus
        config.num_gpus = args.num_gpus
        config.training.epochs = args.epochs
        config.topology.num_nodes = args.num_nodes
        config.attack.byzantine_fraction = args.byzantine_fraction
        config.data_dir = Path(args.data_dir)
        config.results_dir = Path(args.results_dir)
    
    # Create projector
    print("\nInitializing weight projector...")
    base_model = ResNet18_CIFAR()
    projector = RandomWeightProjector.from_model(
        base_model,
        projection_dim=config.projection_dim,
        random_state=args.seed,
    )
    print(f"✓ Projector created: {base_model.get_num_parameters():,} params → "
          f"{config.projection_dim}D")
    
    # Select Byzantine nodes
    bad_indices = select_byzantine_nodes(
        num_nodes=config.topology.num_nodes,
        byzantine_fraction=config.attack.byzantine_fraction,
        strategy="random",
        seed=args.seed,
    )
    
    # Create topology
    topology = create_topology_matrix(
        num_nodes=config.topology.num_nodes,
        topology_type=args.topology,
        topology_file=args.topology_file,
        avg_degree=6,
        seed=args.seed,
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
        batch_size=config.training.batch_size,
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