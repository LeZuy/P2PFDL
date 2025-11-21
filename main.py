# main.py (refactored)
"""Entry point using refactored core module."""

import argparse
import numpy as np
from pathlib import Path

import torch

from decen_learn.core import Node, ByzantineNode, RandomWeightProjector
from decen_learn.aggregators import get_aggregator
from decen_learn.attacks import MinMaxAttack
from decen_learn.config import ExperimentConfig
from model.ResNet_Cifar import ResNet18_CIFAR
from model.train import get_trainloader, get_testloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus", type=str, default="mean")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def create_nodes(config: ExperimentConfig, projector):
    """Create network of nodes."""
    # Load Byzantine indices
    bad_path = Path("./configs/bad_clients_rand.txt")
    if bad_path.exists():
        bad_indices = np.loadtxt(bad_path, dtype=int, ndmin=1).tolist()
    else:
        # Generate random Byzantine nodes
        num_byz = int(config.topology.num_nodes * config.attack.byzantine_fraction)
        bad_indices = np.random.choice(
            config.topology.num_nodes,
            num_byz,
            replace=False
        ).tolist()
    
    print(f"Byzantine nodes: {len(bad_indices)}/{config.topology.num_nodes}")
    
    # Create attack
    attack = MinMaxAttack(boosting_factor=1.0)
    
    # Create nodes
    nodes = []
    for node_id in range(config.topology.num_nodes):
        model = ResNet18_CIFAR()
        dataloader = get_trainloader(
            f"{config.data_dir}/client_{node_id}",
            batch_size=config.training.batch_size,
        )
        
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
    
    return nodes


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config))
    else:
        config = ExperimentConfig()
        config.consensus_type = args.consensus
        config.num_gpus = args.num_gpus
    
    # Create projector
    base_model = ResNet18_CIFAR()
    projector = RandomWeightProjector.from_model(
        base_model,
        projection_dim=config.projection_dim,
        random_state=42,
    )
    
    # Create nodes
    nodes = create_nodes(config, projector)
    
    # Assign topology
    topology = np.loadtxt("./configs/erdos_renyi.txt")
    for node in nodes:
        neighbors = np.nonzero(topology[node.id])[0]
        node.neighbors = neighbors.tolist()
    
    # Create aggregator
    num_byzantine = sum(1 for n in nodes if n.is_byzantine)
    aggregator = get_aggregator(
        config.consensus_type,
        num_byzantine=num_byzantine,
    )
    
    print(f"Starting training with {aggregator.__class__.__name__}")
    
    # Import and use existing training loop
    from simulation.training import run_training
    
    run_training(
        consensus_type=config.consensus_type,
        processes=nodes,
        epochs=config.training.epochs,
        max_parallel_gpu=config.num_gpus,
        results_dir=config.results_dir / config.consensus_type,
    )


if __name__ == "__main__":
    main()