"""Entry point for running decentralized consensus training experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import numpy as np

from model.ResNet_Cifar import ResNet18_CIFAR
from simulation.process import ByzantineNode, Node
from simulation.training import run_training
from utils import (
    build_projection_mats,
    load_projection_mats,
    save_model,
    save_projection_mats,
)


DEFAULT_NUM_PROCESSES = 64
DEFAULT_PROJECTION_DIM = 2
DEFAULT_PROJECTION_CACHE = Path("./configs/projection_mats.npz")
RESULTS_ROOT = Path("./results")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Distributed Training with Different Consensus Methods"
    )
    parser.add_argument(
        "--consensus",
        type=str,
        default="mean",
        choices=["mean", "trimmed_mean", "tverberg", "krum", "geomedian"],
        help="Consensus rule applied during distributed training.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of CUDA devices to distribute clients across.",
    )
    parser.add_argument(
        "--projection-cache",
        type=str,
        default=str(DEFAULT_PROJECTION_CACHE),
        help=(
            "Path to .npz cache of per-layer projection matrices. "
            "Pass an empty string to disable caching."
        ),
    )
    return parser.parse_args()


def load_bad_clients(path: Path) -> List[int]:
    """Return the list of Byzantine client ids configured for the run."""

    raw_ids = np.loadtxt(path, dtype=int)
    return np.atleast_1d(raw_ids).astype(int).tolist()


def build_processes(
    num_processes: int,
    projection_mats: Dict[str, np.ndarray],
    bad_clients: Iterable[int],
    devices: Sequence[torch.device],
) -> List[Node]:
    """Instantiate honest and Byzantine processes with shared projections."""
    bad_client_set = set(bad_clients)

    processes: List[Node] = []
    if not devices:
        raise ValueError("Device list must contain at least one entry.")

    for pid in range(num_processes):
        model_instance = ResNet18_CIFAR()
        device = devices[pid % len(devices)]
        if pid in bad_client_set:
            processes.append(
                ByzantineNode(
                    pid,
                    projection_mats,
                    bad_clients,
                    model_instance,
                    device=device,
                )
            )
        else:
            processes.append(
                Node(
                    pid,
                    projection_mats,
                    model_instance,
                    device=device,
                )
            )
    return processes


def assign_neighbors(processes: List[Node], topology: np.ndarray) -> None:
    """Populate neighbor lists based on the adjacency matrix."""
    for process in processes:
        neighbors = np.nonzero(topology[process.id])[0]
        process.neighbors = [int(node_id) for node_id in neighbors]


def clear_results_dir(path: Path) -> None:
    """Ensure the results directory exists and is empty before training."""
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def main() -> None:
    args = parse_args()
    consensus_type = args.consensus

    num_processes = DEFAULT_NUM_PROCESSES
    projection_dim = DEFAULT_PROJECTION_DIM

    cache_arg = args.projection_cache.strip()
    projection_cache: Optional[Path] = (
        Path(cache_arg).expanduser() if cache_arg else None
    )

    available_gpus = torch.cuda.device_count()
    requested_gpus = max(0, int(args.num_gpus))
    active_gpu_count = min(requested_gpus, available_gpus)

    if active_gpu_count > 0:
        devices = [torch.device(f"cuda:{idx}") for idx in range(active_gpu_count)]
    else:
        devices = [torch.device("cpu")]

    bad_clients_path = Path("./configs/bad_clients_rand.txt")
    bad_clients = load_bad_clients(bad_clients_path)
    # bad_clients = []

    percentage = len(bad_clients) / num_processes * 100
    print(f"Byzantine nodes: {len(bad_clients)} ({percentage:.2f}%) -> {bad_clients}")

    projection_mats = None
    if projection_cache and projection_cache.exists():
        print(f"Loading projection matrices from {projection_cache}")
        projection_mats = load_projection_mats(projection_cache)

    if projection_mats is None:
        print("Building new projection matrices...")
        base_model = ResNet18_CIFAR()
        projection_mats = build_projection_mats(base_model, projection_dim)
        if projection_cache:
            projection_cache.parent.mkdir(parents=True, exist_ok=True)
            save_projection_mats(projection_mats, projection_cache)
            print(f"Saved projection matrices to {projection_cache}")

    processes = build_processes(
        num_processes,
        projection_mats,
        bad_clients,
        devices,
    )

    topology = np.loadtxt("./configs/erdos_renyi.txt")
    assign_neighbors(processes, topology)

    results_dir = RESULTS_ROOT / consensus_type
    # results_dir = Path("./results/optimal")
    clear_results_dir(results_dir)

    max_parallel_gpu = active_gpu_count if active_gpu_count > 0 else 1
    run_training(
        consensus_type,
        processes,
        epochs=200,
        max_parallel_gpu=max_parallel_gpu,
        results_dir=results_dir,
    )
    save_model(processes, path=str(results_dir / "final_models.pth"))

if __name__ == "__main__":
    main()
