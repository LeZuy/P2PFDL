"""Device management for efficient GPU usage."""

import torch
from typing import Optional, Iterable, List
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device allocation and model movement for nodes.
    
    Handles:
    - GPU assignment
    - Model migration between CPU/GPU
    - Memory management
    """
    
    def __init__(
        self,
        node_id: int,
        requested_device: Optional[torch.device] = None,
        tracked_models: Optional[Iterable[torch.nn.Module]] = None,
    ):
        """Initialize device manager.
        
        Args:
            node_id: Node identifier for GPU assignment
            requested_device: Explicitly requested device (optional)
        """
        self.node_id = node_id
        self.assigned_device = self._resolve_device(requested_device)
        self._current_device = torch.device("cpu")
        self._tracked_models: List[torch.nn.Module] = []
        if tracked_models:
            for model in tracked_models:
                self.track(model)
        
        logger.info(
            f"[Node {node_id}] Device manager initialized. "
            f"Assigned: {self.assigned_device}"
        )
    
    def track(self, model: torch.nn.Module) -> None:
        """Register a model so release_all() can move it off GPU later."""
        if model not in self._tracked_models:
            self._tracked_models.append(model)

    def _resolve_device(
        self,
        requested: Optional[torch.device]
    ) -> torch.device:
        """Determine appropriate device for this node."""
        if requested is not None:
            device = torch.device(requested)
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    f"[Node {self.node_id}] CUDA requested but unavailable. "
                    "Falling back to CPU."
                )
                return torch.device("cpu")
            return device
        
        # Auto-assign GPU if available
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_id = self.node_id % gpu_count
            return torch.device(f"cuda:{gpu_id}")
        
        return torch.device("cpu")
    
    def acquire(self, model: torch.nn.Module) -> torch.device:
        """Move model to assigned device for computation.
        
        Args:
            model: Model to move
            
        Returns:
            The device the model was moved to
        """
        target = self.assigned_device
        if self._current_device == target:
            return target
        
        model.to(target)
        self._move_optimizer_state(model, target)
        self._current_device = target
        
        if target.type == "cuda":
            torch.cuda.set_device(target)
        
        return target
    
    def release(self, model: torch.nn.Module) -> None:
        """Move model back to CPU to free GPU memory.
        
        Args:
            model: Model to move
        """
        if self.assigned_device.type != "cuda":
            return  # No need to release CPU
        
        if self._current_device == torch.device("cpu"):
            return  # Already on CPU
        torch.cuda.synchronize(self._current_device)
        model.to("cpu")
        self._move_optimizer_state(model, torch.device("cpu"))
        self._current_device = torch.device("cpu")
        # torch.cuda.empty_cache()  

    def release_all(self) -> None:
        """Release every tracked model back to CPU."""
        for model in list(self._tracked_models):
            self.release(model)
    
    def _move_optimizer_state(
        self,
        model: torch.nn.Module,
        target: torch.device
    ) -> None:
        """Move optimizer state tensors to target device.
        
        This is necessary because optimizer maintains running averages
        that need to be on the same device as the model.
        """
        # Try to find optimizer in model's attributes
        optimizer = getattr(model, '_optimizer', None)
        if optimizer is None:
            return
        
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target, non_blocking=True)
    
    def get_current_device(self) -> torch.device:
        """Return the current device the model is on."""
        return self._current_device
    
    def get_assigned_device(self) -> torch.device:
        """Return the assigned compute device."""
        return self.assigned_device
    
    def is_on_gpu(self) -> bool:
        """Check if currently using GPU."""
        return self._current_device.type == "cuda"
