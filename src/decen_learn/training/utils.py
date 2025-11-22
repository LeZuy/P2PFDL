# src/decen_learn/training/utils.py
"""Training utilities and helper functions."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch


def assign_topology(nodes: List, adjacency_matrix: np.ndarray) -> None:
    """Assign neighbors to nodes based on adjacency matrix.
    
    Args:
        nodes: List of nodes
        adjacency_matrix: Binary adjacency matrix (n x n)
    """
    for node in nodes:
        neighbors = np.nonzero(adjacency_matrix[node.id])[0]
        node.neighbors = neighbors.tolist()
    
    print(f"Assigned topology to {len(nodes)} nodes")
    print(f"  Average degree: {adjacency_matrix.sum(axis=1).mean():.2f}")


def load_topology(path: Path) -> np.ndarray:
    """Load topology from file.
    
    Args:
        path: Path to adjacency matrix file
        
    Returns:
        Adjacency matrix as numpy array
    """
    adj = np.loadtxt(path)
    print(f"Loaded topology from {path}")
    print(f"  Size: {adj.shape[0]} nodes")
    print(f"  Edges: {int(adj.sum())} directed")
    return adj


def create_topology_erdos_renyi(
    num_nodes: int,
    avg_degree: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create Erdős-Rényi random graph topology.
    
    Args:
        num_nodes: Number of nodes
        avg_degree: Target average degree
        seed: Random seed
        
    Returns:
        Adjacency matrix
    """
    import networkx as nx
    
    num_edges = int(num_nodes * avg_degree / 2)
    
    # Keep generating until minimum degree constraint is met
    rng = np.random.default_rng(seed)
    graph_seed = seed if seed is not None else rng.integers(0, 2**31)
    
    for attempt in range(100):
        G = nx.gnm_random_graph(num_nodes, num_edges, seed=graph_seed)
        adj = nx.to_numpy_array(G, dtype=int)
        
        # Check minimum degree (at least 2)
        degrees = adj.sum(axis=1)
        if (degrees >= 2).all():
            print(f"Created Erdős-Rényi topology (attempt {attempt + 1})")
            print(f"  Nodes: {num_nodes}, Edges: {int(adj.sum())}")
            print(f"  Degree: min={int(degrees.min())}, "
                  f"avg={degrees.mean():.2f}, max={int(degrees.max())}")
            return adj
        
        graph_seed += 1
    
    raise RuntimeError(
        f"Failed to create valid topology after 100 attempts. "
        f"Try increasing avg_degree."
    )


def create_topology_ring(num_nodes: int, degree: int) -> np.ndarray:
    """Create ring lattice topology.
    
    Args:
        num_nodes: Number of nodes
        degree: Number of neighbors per node
        
    Returns:
        Adjacency matrix
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(1, degree + 1):
            neighbor = (i + j) % num_nodes
            adj[i, neighbor] = 1
            adj[neighbor, i] = 1  # Undirected
    
    print(f"Created ring lattice topology")
    print(f"  Nodes: {num_nodes}, Degree: {degree}")
    
    return adj


def select_byzantine_nodes(
    num_nodes: int,
    byzantine_fraction: float,
    strategy: str = "random",
    seed: Optional[int] = None,
    adjacency_matrix: Optional[np.ndarray] = None,
) -> List[int]:
    """Select which nodes should be Byzantine.
    
    Args:
        num_nodes: Total number of nodes
        byzantine_fraction: Fraction of Byzantine nodes (0 to 1)
        strategy: Selection strategy ("random", "evenly_spaced", "high_degree")
        seed: Random seed for reproducibility
        adjacency_matrix: Required for "high_degree" strategy
        
    Returns:
        List of Byzantine node indices
    """
    num_byzantine = int(num_nodes * byzantine_fraction)
    rng = np.random.default_rng(seed)
    
    if strategy == "random":
        bad_indices = rng.choice(num_nodes, num_byzantine, replace=False)
    
    elif strategy == "evenly_spaced":
        # Select evenly spaced indices
        if num_byzantine == 0:
            bad_indices = np.array([], dtype=int)
        else:
            step = num_nodes // num_byzantine
            bad_indices = np.array([
                (k * step) % num_nodes for k in range(num_byzantine)
            ], dtype=int)
    
    elif strategy == "high_degree":
        # Select nodes with highest degree
        if adjacency_matrix is None:
            raise ValueError("adjacency_matrix required for high_degree strategy")
        degrees = adjacency_matrix.sum(axis=1)
        bad_indices = np.argsort(degrees)[-num_byzantine:]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    bad_indices = sorted(bad_indices.tolist())
    
    print(f"Selected {len(bad_indices)} Byzantine nodes ({strategy}): {bad_indices}")
    
    return bad_indices


def verify_byzantine_constraint(
    adjacency_matrix: np.ndarray,
    bad_nodes: List[int],
    max_bad_neighbors_fraction: float = 1/3,
) -> bool:
    """Verify that no node has too many Byzantine neighbors.
    
    For Tverberg/Krum to work, each honest node should have at most
    1/3 of its neighbors be Byzantine.
    
    Args:
        adjacency_matrix: Network topology
        bad_nodes: List of Byzantine node indices
        max_bad_neighbors_fraction: Maximum allowed fraction
        
    Returns:
        True if constraint is satisfied
    """
    n = adjacency_matrix.shape[0]
    bad_mask = np.zeros(n, dtype=bool)
    bad_mask[bad_nodes] = True
    
    # Count bad neighbors for each node
    bad_neighbor_counts = np.sum(adjacency_matrix[:, bad_mask] != 0, axis=1)
    total_neighbors = np.sum(adjacency_matrix != 0, axis=1)
    
    # Check constraint
    violations = []
    for i in range(n):
        if total_neighbors[i] == 0:
            continue
        fraction = bad_neighbor_counts[i] / total_neighbors[i]
        if fraction > max_bad_neighbors_fraction + 1e-6:
            violations.append((i, fraction, bad_neighbor_counts[i]))
    
    if violations:
        print(f"⚠️  Byzantine constraint violated for {len(violations)} nodes:")
        for node_id, frac, count in violations:
            print(f"\tNode {node_id}: {count}/{int(total_neighbors[node_id])} "
                  f"= {frac:.2%} bad neighbors")
        return False
    
    print(f"✓ Byzantine constraint satisfied (max {max_bad_neighbors_fraction:.1%})")
    return True


def save_training_metadata(
    nodes: List,
    config,
    path: Path,
) -> None:
    """Save training metadata for reproducibility.
    
    Args:
        nodes: List of nodes
        config: Experiment configuration
        path: Output path
    """
    import json
    
    byzantine_ids = [n.id for n in nodes if n.is_byzantine]
    
    metadata = {
        'num_nodes': len(nodes),
        'num_byzantine': len(byzantine_ids),
        'byzantine_ids': byzantine_ids,
        'consensus_type': config.consensus_type,
        'epochs': config.training.epochs,
        'batch_size': config.training.batch_size,
        'learning_rate': config.training.learning_rate,
        'projection_dim': config.projection_dim,
    }
    
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved training metadata to {path}")


def load_checkpoint(
    path: Path,
    nodes: List,
) -> int:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        nodes: List of nodes to restore
        
    Returns:
        Epoch number to resume from
    """
    checkpoint = torch.load(path)
    
    for node in nodes:
        key = f"node_{node.id}"
        if key in checkpoint:
            node.model.load_state_dict(checkpoint[key])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    
    return epoch


def save_checkpoint(
    path: Path,
    nodes: List,
    epoch: int,
    metrics: Optional[dict] = None,
) -> None:
    """Save training checkpoint.
    
    Args:
        path: Output path
        nodes: List of nodes
        epoch: Current epoch
        metrics: Optional metrics dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics or {},
    }
    
    for node in nodes:
        checkpoint[f"node_{node.id}"] = node.model.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def print_training_summary(
    nodes: List,
    config,
) -> None:
    """Print training configuration summary.
    
    Args:
        nodes: List of nodes
        config: Experiment configuration
    """
    num_byzantine = sum(1 for n in nodes if n.is_byzantine)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Experiment: {config.name}")
    print(f"Consensus: {config.consensus_type}")
    print(f"\nNetwork:")
    print(f"\tTotal nodes: {len(nodes)}")
    print(f"\tHonest: {len(nodes) - num_byzantine}")
    print(f"\tByzantine: {num_byzantine} ({100*num_byzantine/len(nodes):.1f}%)")
    print(f"\nTraining:")
    print(f"\tEpochs: {config.training.epochs}")
    print(f"\tBatch size: {config.training.batch_size}")
    print(f"\tLearning rate: {config.training.learning_rate}")
    print(f"\tConsensus interval: {config.training.consensus_interval}")
    print(f"\nResources:")
    print(f"\tGPUs available: {torch.cuda.device_count()}")
    print(f"\tMax parallel: {config.num_gpus}")
    print(f"\nOutput:")
    print(f"\tResults dir: {config.results_dir / config.consensus_type}")
    print("="*60 + "\n")