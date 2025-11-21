import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmattack as ma
import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import DataLoader
import yaml

from aggregator import krum, geo_median, trimmed_mean
from gilberts import gilbert_coefficients
from model.train import get_trainloader, training_step
from tverberg import centerpoint_2d
from utils import (
    flatten_weights,
    is_point_in_convex_hull,
    get_projection_matrix,
    model_params,
    random_project,
    restore_state_dict,
)
from ransac import ransac_simplex
from ipm import craft_ipm_local

__all__ = ["Node", "ByzantineNode"]

VectorLike = Union[np.ndarray, torch.Tensor]
WeightEntry = Tuple[VectorLike, Tuple[int, ...]]
LayerWeights = Dict[str, WeightEntry]
WeightBuffer = List[LayerWeights]


class Node:
    """Represents an honest client participating in decentralized training."""

    PARAMS_PATH = "./configs/params.yaml"

    def __init__(
        self,
        pid: int,
        Pi_dict: Dict[str, np.ndarray],
        model: torch.nn.Module,
        data_dir: str = "./data_splits",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        self.id = pid
        self.is_byzantine = False
        self.Pi = Pi_dict
        self.neighbors: List[int] = []
        self.B: WeightBuffer = []
        self.B_proj: WeightBuffer = []

        self.device = self._resolve_device(pid, device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        print(f"[Node {self.id}] Using device: {self.device}")

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.weights: LayerWeights = self._normalize_weights(
            model_params(self.model)
        )
        # Cache layer shapes and flat dimension for fast flatten/unflatten
        self.layer_shapes = [
            (name, len(vec), shape) for name, (vec, shape) in self.weights.items()
        ]
        self.flat_dim = sum(size for _, size, _ in self.layer_shapes)
        self.P_flat = Pi_dict.get("__flat__")
        if self.P_flat is None or self.P_flat.shape[1] != self.flat_dim:
            # Build a shared projection for the full model vector when missing
            self.P_flat = get_projection_matrix(self.flat_dim, 2)
            self.Pi["__flat__"] = self.P_flat

        self.proj_weights: LayerWeights = self.project(self.weights)
        self.gb_coef: Dict[str, np.ndarray] = {}
        self.loss: float = 0.0
        self.trainloader = get_trainloader(f"{data_dir}/client_{self.id}", batch_size=64)
        self._model_device = torch.device("cpu")
        self._cpu_device = torch.device("cpu")

    # ---------- Helpers ----------
    @staticmethod
    def _resolve_device(
        pid: int, requested: Optional[Union[str, torch.device]]
    ) -> torch.device:
        """Select the execution device for the current node."""
        if requested is not None:
            device = torch.device(requested)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but no GPU available")
            return device

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            return torch.device(f"cuda:{pid % gpu_count}")
        return torch.device("cpu")

    @staticmethod
    def _to_numpy(vector: VectorLike) -> np.ndarray:
        if isinstance(vector, torch.Tensor):
            return vector.detach().to("cpu").numpy()
        return np.asarray(vector)

    @classmethod
    def _normalize_weights(
        cls, weights: Dict[str, Sequence[Any]]
    ) -> LayerWeights:
        """Convert raw model parameters into numpy arrays with shape metadata."""
        normalized: LayerWeights = {}
        for name, value in weights.items():
            vector, shape = value
            normalized[name] = (cls._to_numpy(vector), tuple(shape))
        return normalized

    def _flatten_weights(self, weights: LayerWeights) -> np.ndarray:
        """Flatten LayerWeights dict into a single 1-D vector."""
        return flatten_weights(weights)

    def _unflatten_vector(self, flat: VectorLike) -> LayerWeights:
        """Reconstruct LayerWeights from a flattened vector using cached shapes."""
        flat_arr = self._to_numpy(flat).reshape(-1)
        restored: LayerWeights = {}
        pos = 0
        for name, size, shape in self.layer_shapes:
            segment = flat_arr[pos : pos + size]
            restored[name] = (segment.copy(), shape)
            pos += size
        return restored

    def _move_optimizer_state(self, target: torch.device) -> None:
        """Move optimizer state tensors to the target device."""
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target, non_blocking=True)

    def _ensure_model_device(self, target: torch.device) -> torch.device:
        """Ensure the model, criterion, and optimizer state live on target."""
        if self._model_device == target:
            return self._model_device
        self.model.to(target)
        self.criterion.to(target)
        self._move_optimizer_state(target)
        self._model_device = target
        return self._model_device

    def _acquire_device(self) -> torch.device:
        """Bring model to its assigned device for computation."""
        target = self.device if self.device.type == "cuda" else self._cpu_device
        return self._ensure_model_device(target)

    def _release_device(self) -> None:
        """Release GPU memory by moving model back to CPU when idle."""
        if self.device.type != "cuda":
            return
        self._ensure_model_device(self._cpu_device)

    # ---------- Local training ----------
    def train_one_epoch(self) -> None:
        """Run one local training epoch and refresh cached weights."""
        train_device = self._acquire_device()
        with open(self.PARAMS_PATH, "r", encoding="utf-8") as handle:
            params_loaded = yaml.safe_load(handle)
        self.loss = training_step(
            self.model,
            self.trainloader,
            self.criterion,
            self.optimizer,
            train_device,
            params=params_loaded,
            is_bad=False,
        )
        self.scheduler.step()
        self.weights = self._normalize_weights(model_params(self.model))
        self.proj_weights = self.project(self.weights)
        self._release_device()

    # ---------- Projection ----------
    def project(self, weights: LayerWeights) -> LayerWeights:
        """Project the flattened model vector using a shared projection."""
        flat_vec = flatten_weights(weights)
        projected = random_project(flat_vec, self.P_flat)
        return {"__flat__": (projected, (self.flat_dim,))}

    def project_B(self) -> None:
        self.B_proj = [self.project(buffer) for buffer in self.B]

    # ---------- Consensus ----------
    def reset_B(self) -> None:
        """Reset buffer before consensus; start empty to store only neighbors."""
        self.B = []
        self.B_proj = []
        # Clear convex coefficients so Tverberg preimage is recomputed each round
        self.gb_coef = {}

    def prepare_broadcast(
        self,
        consensus_type: str,
        processes: Sequence["Node"],
    ) -> LayerWeights:
        """Return the weight payload shared with neighbors."""
        _ = consensus_type  # kept for API symmetry
        _ = processes
        return copy.deepcopy(self.weights)

    def send_v(self, other: "Node") -> None:
        """Send a deep copy of local weights to another process."""
        other.B.append(copy.deepcopy(self.weights))

    def consensus(self, consensus_type: str) -> LayerWeights:
        """Aggregate projected weights using the specified consensus method."""
        if not self.B_proj:
            # Nothing received; skip aggregation and keep local weights.
            self.proj_weights = self.project(self.weights)
            return copy.deepcopy(self.weights)

        consensus_type = consensus_type.lower()
        num_vectors = len(self.B_proj)

        # Always aggregate on the flattened model vector
        vectors = np.stack(
            [
                self._to_numpy(
                    buffer["__flat__"][0] if "__flat__" in buffer else flatten_weights(buffer)
                )
                for buffer in self.B_proj
            ],
            axis=0,
        )

        if consensus_type == "krum":
            return self._krum_consensus(vectors)

        if consensus_type == "mean":
            v_bar = np.mean(vectors, axis=0)
        elif consensus_type == "geomedian":
            v_bar = geo_median(vectors)
        elif consensus_type == "trimmed_mean":
            num_attackers = max(1, int(np.floor(num_vectors / 3)))
            v_bar = trimmed_mean(vectors, num_attackers)
        elif consensus_type == "tverberg":
            v_bar, _ = centerpoint_2d(vectors)
            if not is_point_in_convex_hull(v_bar, vectors):
                print(
                    f"Tverberg point not in convex hull (Node {self.id})"
                )
                np.savetxt(
                    f"./debugs/node{self.id}_flatten_failed.txt",
                    np.vstack([vectors, v_bar]),
                    fmt="%.6f",
                )
        else:
            raise ValueError(
                f"Unsupported consensus_type '{consensus_type}'."
            )

        # Store projected weights (for logging) and return unflattened dict
        self.proj_weights = {"__flat__": (v_bar, (self.flat_dim,))}
        consensus_weight_dict = self._unflatten_vector(v_bar)

        return consensus_weight_dict

    def _krum_consensus(self, vectors: np.ndarray) -> LayerWeights:
        """Select a single neighbor payload using Krum on flattened weights."""
        num_vectors = vectors.shape[0]
        est_attackers = int(np.floor(num_vectors / 3))
        max_valid_f = max(0, (num_vectors - 3) // 2)
        num_attackers = max(0, min(est_attackers, max_valid_f))

        if num_attackers <= 0:
            # Not enough vectors for Krum; fall back to a simple mean.
            v_bar = np.mean(vectors, axis=0)
            self.proj_weights = {"__flat__": (v_bar, (self.flat_dim,))}
            return self._unflatten_vector(v_bar)

        selected_idx, selected_vec = krum(
            vectors,
            num_attackers,
            return_index=True,
        )
        self.proj_weights = {"__flat__": (selected_vec, (self.flat_dim,))}
        return copy.deepcopy(self.B[selected_idx])

    # ---------- Update weight ----------
    def get_cvx_coef(self) -> Dict[str, np.ndarray]:
        """Compute coefficients for the current projected consensus."""
        
        coefficients: Dict[str, np.ndarray] = {}
        for layer, (proj_vec, _) in self.proj_weights.items():
            vectors = np.stack(
                [self._to_numpy(buffer[layer][0]) for buffer in self.B_proj],
                axis=0,
            )

            target = self._to_numpy(proj_vec)
            rng = np.random.default_rng(0)

            res = ransac_simplex(
                vectors, q=target, mode="contain_q",
                iterations=int(1e6), eps=1e-9, rng=rng, early_stop_rounds=1000)
            coff = res.get('q_weights_dense')

            coefficients[layer] = coff
        self.gb_coef = coefficients
        return coefficients

    def restore_weights_preimage(self) -> LayerWeights:
        """Lift the consensus solution back to the original parameter space."""
        if not self.gb_coef:
            self.get_cvx_coef()

        stacked = np.stack(
            [
                self._to_numpy(
                    self._flatten_weights(buffer)
                    if "__flat__" not in buffer
                    else buffer["__flat__"][0]
                )
                for buffer in self.B
            ],
            axis=0,
        )
        alpha = np.asarray(self.gb_coef.get("__flat__")).reshape(1, -1)
        restored_flat = alpha @ stacked
        return self._unflatten_vector(restored_flat.flatten())

    def update_weights(self, w_plus: LayerWeights, lamb: float = 0.0) -> None:
        """Blend local weights with the consensus solution and reload the model."""
        # Perform the update on the flattened vector to avoid layer-by-layer drift.
        current_flat = self._flatten_weights(self.weights)
        if "__flat__" in w_plus:
            target_flat = self._to_numpy(w_plus["__flat__"][0]).reshape(-1)
        else:
            target_flat = self._flatten_weights(w_plus)
        updated_flat = lamb * current_flat + (1 - lamb) * target_flat
        self.weights = self._unflatten_vector(updated_flat)
        self.model.load_state_dict(restore_state_dict(self.model, self.weights))

    # ---------- Evaluation ----------
    @torch.no_grad()
    def test(
        self,
        testloader: Optional[DataLoader] = None,
        target_class: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Evaluate the current model on the provided dataloader.

        Returns:
            A tuple of (avg_loss, accuracy, attack_success_rate).
        """
        if testloader is None:
            raise ValueError("testloader must be provided for evaluation.")

        eval_device = self._acquire_device()
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0
        total_triggered, hit_target = 0, 0

        for inputs, targets in testloader:
            inputs, targets = inputs.to(eval_device), targets.to(eval_device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss_total += loss.detach().item() * targets.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if target_class is not None:
            #     total_triggered += targets.size(0)
            #     hit_target += (predicted == target_class).sum().item()

        avg_loss = loss_total / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        asr = hit_target / total_triggered if total_triggered > 0 else 0.0
        self._release_device()

        return avg_loss, acc, asr


class ByzantineNode(Node):
    """Node that crafts adversarial updates instead of honest ones."""

    def __init__(
        self,
        pid: int,
        Pi_dict: Dict[str, np.ndarray],
        bad_clients: Sequence[int],
        model: torch.nn.Module,
        data_dir: str = "./data_splits",
        attack_std: Optional[float] = None,
        attack_multiplier: float = 5.0,
        rng_seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(pid, Pi_dict, model, data_dir=data_dir, device=device)
        self.is_byzantine = True
        self.bad_clients = set(bad_clients)
        self.attack_std = attack_std
        self.attack_multiplier = attack_multiplier
        self.boosting_factor = 1.0
        self.rng = default_rng(rng_seed) if rng_seed is not None else None

    def update_weights(self, w_plus: LayerWeights, lamb: float = 0.8) -> None:
        """Byzantine nodes skip honest updates to preserve malicious state."""

    def prepare_broadcast(
        self,
        consensus_type: str,
        processes: Sequence["Node"],
    ) -> LayerWeights:
        """Craft and return the malicious payload that neighbors will receive."""
        neighbor_buffers: WeightBuffer = []
        neighbor_mask: List[bool] = []
        for n_id in self.neighbors:
            neighbor = processes[n_id]
            if neighbor.is_byzantine:
                continue
            neighbor_buffers.append(copy.deepcopy(neighbor.weights))
            neighbor_mask.append(True)
        neighbor_buffers.append(copy.deepcopy(self.weights))
        neighbor_mask.append(False)  # attacker itself
        payload = self._craft_malicious_weights(
            neighbor_buffers,
            consensus_type=consensus_type,
            good_mask=neighbor_mask,
        )
        return payload

    def send_v(self, other: "Node") -> None:
        """Send a deep copy of local weights to another process."""
        bad_idcs = np.loadtxt(np.loadtxt("./configs/bad_clients_rand.txt"))
        if other.id not in bad_idcs:
            other.B.append(copy.deepcopy(self.weights))

    def restore_weights_preimage(self) -> LayerWeights:
        return self.weights

    def consensus(self, consensus_type: str) -> LayerWeights:
        """Byzantine nodes keep their internal weights unchanged."""
        _ = consensus_type
        return self.weights

    def _craft_malicious_weights(
        self,
        buffers: WeightBuffer,
        *,
        consensus_type: str,
        good_mask: Optional[Sequence[bool]] = None,
        oracle_type: str = "minmax",
        tau: float = 1e-3,
        gamma_init: float = 20.0,
    ) -> LayerWeights:
        """Use mmattack to craft adversarial vectors given local buffers."""
        mask = good_mask if good_mask is not None else [True] * len(buffers)

        # Stack full flattened model vectors to craft a single malicious update.
        flat_vectors = torch.stack(
            [
                torch.as_tensor(
                    self._to_numpy(
                        buffer["__flat__"][0]
                        if "__flat__" in buffer
                        else self._flatten_weights(buffer)
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
                for buffer in buffers
            ],
            dim=0,
        )
        v_m, _, _, _ = ma.craft_malicious_vector(
                vectors=flat_vectors,
                consensus_type=consensus_type,
                oracle_type=oracle_type,
                gamma_init=gamma_init,
                tau=tau,
                perturb_kind="auto",
            )
        # v_m = craft_ipm_local(vectors=flat_vectors, good_mask=mask, eps=0.5)
        malicious_flat = self.boosting_factor * v_m.detach().to("cpu").numpy()
        return self._unflatten_vector(malicious_flat)
