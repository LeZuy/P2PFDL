# simulation/training.py
"""Core training loop for decentralized consensus experiments."""

from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from model.train import get_testloader
from utils import save_projected_model, save_consensus_payloads


# Snapshot projected weights every N epochs (plus the warm-up window).
PROJECTION_SNAPSHOT_INTERVAL = 20
EARLY_PROJECTION_EPOCHS = 10


def run_training(
    consensus_type: str,
    processes: Sequence,
    epochs: int,
    *,
    consensus_interval: int = 1,
    test_interval: int = 5,
    max_parallel_gpu: int = 1,
    results_dir: Path | None = None,
) -> None:
    """Run the distributed training loop.

    Args:
        consensus_type: Name of the consensus strategy (e.g. ``"tverberg"``).
        processes: Ordered collection of participating nodes.
        epochs: Number of global epochs to execute.
        consensus_interval: Frequency (in epochs) for triggering consensus.
        test_interval: Frequency (in epochs) for running evaluations.
        max_parallel_gpu: Maximum number of GPU-backed clients to run concurrently
            during local training and evaluation.
        results_dir: Optional directory where intermediate metrics are saved. Falls
            back to ``./results/<consensus_type>`` when unspecified.
    """

    num_processes = len(processes)
    loss = np.zeros(num_processes, dtype=float)
    if results_dir is None:
        results_dir = Path("./results") / consensus_type
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    clean_loader = get_testloader("./data_splits/test/", batch_size=64)
    is_tverberg = consensus_type == "tverberg"

    for epoch in range(epochs):
        _run_local_training(processes, loss, max_parallel_gpu)

        if (epoch + 1) % consensus_interval == 0:
            _run_consensus_phase(
                epoch,
                processes,
                consensus_type,
                is_tverberg,
                results_dir,
            )

        if epoch % test_interval == 0:
            _evaluate_processes(
                epoch,
                processes,
                consensus_type,
                loss,
                results_dir,
                clean_loader,
                max_parallel_gpu,
            )


def _run_local_training(
    processes: Sequence,
    loss: np.ndarray,
    max_parallel_gpu: int,
) -> None:
    """Train each participant for one epoch and refresh loss cache."""

    effective_workers = max(1, int(max_parallel_gpu))
    for group in _chunk_processes(processes, effective_workers):
        if len(group) == 1:
            process = group[0]
            process.train_one_epoch()
            loss[process.id] = process.loss[0]
            continue

        with ThreadPoolExecutor(max_workers=len(group)) as executor:
            futures = {executor.submit(_train_process, proc): proc for proc in group}
            for future, process in futures.items():
                metrics = future.result()
                loss[process.id] = metrics[0]


def _train_process(process):
    """Wrapper to run one local epoch and return cached metrics."""
    process.train_one_epoch()
    return process.loss


def _chunk_processes(processes: Sequence, chunk_size: int):
    """Yield fixed-size chunks while preserving order."""
    for start in range(0, len(processes), chunk_size):
        yield processes[start : start + chunk_size]


def _evaluate_single(process, clean_loader) -> np.ndarray:
    """Evaluate a single process on clean and poisoned loaders."""
    avg_loss, acc, _ = process.test(clean_loader)
    asr = 0.0
    return np.array([avg_loss, acc, asr])


def _group_processes_by_device(processes: Sequence):
    """Bucket processes by their assigned execution device."""
    buckets = {}
    for process in processes:
        device = process.device
        key = (device.type, device.index)
        if key not in buckets:
            buckets[key] = deque()
        buckets[key].append(process)
    return list(buckets.values())


def _round_robin_batches(device_groups, max_parallel_gpu: int):
    """Yield batches that select at most one process per device."""
    limit = max(1, int(max_parallel_gpu))
    while True:
        batch = []
        for group in device_groups:
            if not group:
                continue
            batch.append(group.popleft())
            if len(batch) >= limit:
                break
        if not batch:
            break
        yield batch


def _run_consensus_phase(
    epoch: int,
    processes: Sequence,
    consensus_type: str,
    is_tverberg: bool,
    results_dir: Path,
) -> None:
    print(f"\n=== Consensus phase after Epoch {epoch + 1} ===")
    start_time = time.time()

    for process in processes:
        process.reset_B()

    broadcast_payloads = [
        process.prepare_broadcast(consensus_type, processes)
        for process in processes
    ]

    for process in processes:
        for neighbor_id in process.neighbors:
            sender = processes[neighbor_id]
            if process.is_byzantine and sender.is_byzantine:
                continue
            process.B.append(copy.deepcopy(broadcast_payloads[neighbor_id]))
        # Always include the node's own weights so Krum/other rules can fallback
        # to a self-update when neighbors are scarce or malicious-heavy.
        process.B.append(copy.deepcopy(broadcast_payloads[process.id]))

    if is_tverberg:
        for process in processes:
            process.project_B()
    else:
        for process in processes:
            process.B_proj = process.B

    record_projection =  _should_snapshot_projection(epoch)
    if record_projection:
        save_consensus_payloads(
            processes,
            path=str(
                results_dir / f"payloads_epoch_{epoch + 1}.npy"
            ),
            project_before_saving=not is_tverberg,
        )
        save_projected_model(
            processes,
            2,
            path=str(
                results_dir / f"proj_weights_before_epoch_{epoch + 1}.npy"
            ),
        )

    neighbor_counts = [len(process.B) for process in processes]

    consensus_outputs = [
        process.consensus(consensus_type) for process in processes
    ]

    if record_projection:
        save_projected_model(
            processes,
            2,
            path=str(
                results_dir / f"proj_weights_after_epoch_{epoch + 1}.npy"
            ),
        )

    for process, consensus_output, neighbor_count in zip(
        processes,
        consensus_outputs,
        neighbor_counts,
    ):
        updated_weights = (
            process.restore_weights_preimage()
            if is_tverberg and neighbor_count > 0
            else consensus_output
        )
        # self_weight = 1.0 / (neighbor_count + 1) if neighbor_count > 0 else 1.0
        process.update_weights(updated_weights, lamb=0)

    print(f"Consensus phase took {time.time() - start_time:.2f}s")


def _evaluate_processes(
    epoch: int,
    processes: Sequence,
    consensus_type: str,
    loss: np.ndarray,
    results_dir: Path,
    clean_loader,
    max_parallel_gpu: int,
) -> None:
    print(f"\n--- Test results after Epoch {epoch + 1} ---")
    metrics = np.zeros((len(processes), 3), dtype=float)

    device_groups = _group_processes_by_device(processes)
    for batch in _round_robin_batches(device_groups, max_parallel_gpu):
        if len(batch) == 1:
            process = batch[0]
            metrics[process.id] = _evaluate_single(
                process, clean_loader
            )
            continue

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    _evaluate_single, process, clean_loader
                ): process
                for process in batch
            }
            for future, process in futures.items():
                metrics[process.id] = future.result()

    np.savetxt(
        results_dir / f"loss_epoch_{epoch + 1}.txt",
        loss,
        fmt="%.4f",
    )
    np.savetxt(
        results_dir / f"test_results_epoch_{epoch + 1}.txt",
        metrics,
        fmt="%.4f",
    )


def _should_snapshot_projection(epoch: int) -> bool:
    """Return True when projected buffers should be exported for inspection."""

    return (
        epoch % PROJECTION_SNAPSHOT_INTERVAL == 0
        or epoch < EARLY_PROJECTION_EPOCHS
    )
