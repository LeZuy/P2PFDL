# scripts/example_usage.py
"""Example usage of the refactored core module."""

import torch
import numpy as np
from pathlib import Path

from decen_learn.core import (
    Node,
    ByzantineNode,
    RandomWeightProjector,
)
from decen_learn.aggregators import get_aggregator
from decen_learn.attacks import MinMaxAttack
from model.ResNet_Cifar import ResNet18_CIFAR
from model.train import get_trainloader


def create_network(
    num_nodes: int = 10,
    num_byzantine: int = 3,
    projection_dim: int = 2,
    data_dir: str = "./data_splits",
):
    """Create a network of honest and Byzantine nodes.
    
    Args:
        num_nodes: Total number of nodes
        num_byzantine: Number of Byzantine nodes
        projection_dim: Projection dimensionality
        data_dir: Path to partitioned data
        
    Returns:
        List of nodes
    """
    # Create base model to determine dimensionality
    base_model = ResNet18_CIFAR()
    
    # Create shared projector
    projector = RandomWeightProjector.from_model(
        base_model,
        projection_dim=projection_dim,
        random_state=42,
    )
    
    # Determine Byzantine nodes
    bad_indices = np.random.choice(num_nodes, num_byzantine, replace=False)
    bad_indices = sorted(bad_indices.tolist())
    
    print(f"Byzantine nodes: {bad_indices}")
    
    # Create attack strategy
    attack = MinMaxAttack(
        boosting_factor=1.0,
        gamma_init=20.0,
        tau=1e-3,
    )
    
    # Create nodes
    nodes = []
    for node_id in range(num_nodes):
        model = ResNet18_CIFAR()
        dataloader = get_trainloader(
            f"{data_dir}/client_{node_id}",
            batch_size=64
        )
        
        if node_id in bad_indices:
            # Create Byzantine node
            node = ByzantineNode(
                node_id=node_id,
                model=model,
                projector=projector,
                dataloader=dataloader,
                attack=attack,
                bad_client_ids=bad_indices,
                device=None,  # Auto-assign
            )
        else:
            # Create honest node
            node = Node(
                node_id=node_id,
                model=model,
                projector=projector,
                dataloader=dataloader,
                device=None,  # Auto-assign
            )
        
        nodes.append(node)
    
    return nodes


def assign_topology(nodes, adjacency_matrix):
    """Assign neighbors based on adjacency matrix.
    
    Args:
        nodes: List of nodes
        adjacency_matrix: Adjacency matrix (numpy array)
    """
    for node in nodes:
        neighbors = np.nonzero(adjacency_matrix[node.id])[0]
        node.neighbors = neighbors.tolist()


def run_consensus_round(nodes, aggregator):
    """Execute one round of consensus.
    
    Args:
        nodes: List of nodes
        aggregator: Aggregator instance
    """
    # Reset buffers
    for node in nodes:
        node.reset_buffers()
    
    # Prepare broadcasts
    broadcasts = {}
    for node in nodes:
        if node.is_byzantine:
            # Byzantine nodes need access to all nodes to craft attacks
            broadcasts[node.id] = node.prepare_broadcast(processes=nodes)
        else:
            broadcasts[node.id] = node.prepare_broadcast()
    
    # Exchange messages
    for node in nodes:
        # Receive from neighbors
        for neighbor_id in node.neighbors:
            sender = nodes[neighbor_id]
            # Byzantine nodes don't send to each other
            if node.is_byzantine and sender.is_byzantine:
                continue
            node.receive(broadcasts[neighbor_id])
        
        # Include self (for fallback in sparse networks)
        node.receive(broadcasts[node.id])
    
    # Project buffers (for Tverberg)
    for node in nodes:
        if not node.is_byzantine:
            node.project_buffers()
    
    # Aggregate
    for node in nodes:
        if not node.is_byzantine:
            aggregated = node.aggregate(aggregator)
            node.update_weights(aggregated, momentum=0.0)


def main():
    """Main example execution."""
    print("Creating network...")
    nodes = create_network(
        num_nodes=10,
        num_byzantine=3,
        projection_dim=2,
    )
    
    # Load topology
    topology = np.loadtxt("./configs/erdos_renyi.txt")
    if topology.shape[0] != len(nodes):
        # Use simple ring topology for demo
        topology = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            topology[i, (i + 1) % len(nodes)] = 1
            topology[i, (i - 1) % len(nodes)] = 1
    
    assign_topology(nodes, topology)
    
    # Create aggregator
    num_byzantine = sum(1 for n in nodes if n.is_byzantine)
    aggregator = get_aggregator(
        "tverberg",
        num_byzantine=num_byzantine,
    )
    
    print(f"\nStarting training with {aggregator.__class__.__name__}...")
    
    # Training loop
    for epoch in range(5):
        print(f"\n=== Epoch {epoch + 1} ===")
        
        # Local training
        for node in nodes:
            loss, acc = node.train_epoch()
            if epoch == 0:  # Print first epoch
                node_type = "BYZ" if node.is_byzantine else "HON"
                print(
                    f"[{node_type} {node.id}] "
                    f"Loss: {loss:.4f}, Acc: {acc:.2f}%"
                )
        
        # Consensus
        print("Running consensus...")
        run_consensus_round(nodes, aggregator)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()