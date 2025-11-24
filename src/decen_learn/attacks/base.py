# src/decen_learn/attacks/base.py
"""Base attack class."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from decen_learn.core.utils import (
    flatten_weight_dict,
    unflatten_weight_dict,
)

@dataclass
class AttackResult:
    """Output from attack crafting."""
    malicious_weights: Dict[str, torch.Tensor]
    metadata: dict = field(default_factory=dict)


class BaseAttack(ABC):
    """Abstract base class for Byzantine attacks."""
    
    def __init__(
        self,
        boosting_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.boosting_factor = boosting_factor
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
    
    @abstractmethod
    def craft(
        self,
        honest_weights: List[Dict[str, torch.Tensor]],
        attacker_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Craft malicious weights.
        
        Args:
            honest_weights: List of weight dicts from honest neighbors
            attacker_weights: The attacker's current weights
            
        Returns:
            Malicious weight dict to broadcast
        """
        pass
    
    def set_device(self, device: torch.device) -> None:
        """Update attack's computation device."""
        self.device = torch.device(device)
    
    def to(self, device: torch.device) -> "BaseAttack":
        """Return a copy of this attack configured for the provided device."""
        clone = deepcopy(self)
        clone.set_device(device)
        return clone
    
    def _flatten(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten weight dictionary to 1D vector."""
        return flatten_weight_dict(
            weights,
            detach=True,
            to_device=self.device,
        )
    
    def _unflatten(
        self, 
        flat: torch.Tensor, 
        template: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Restore flattened vector to weight dictionary using template."""
        target_device = self._infer_template_device(template)
        return unflatten_weight_dict(
            flat,
            template,
            clone=False,
            to_device=target_device,
        )
    
    @staticmethod
    def _clone_template(template: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return a detached clone of the provided template dictionary."""
        return {
            name: tensor.detach().clone()
            for name, tensor in template.items()
        }
    
    @staticmethod
    def _infer_template_device(
        template: Dict[str, torch.Tensor]
    ) -> torch.device:
        """Infer appropriate device for reconstructed tensors."""
        for tensor in template.values():
            if torch.is_tensor(tensor):
                return tensor.device
        return torch.device("cpu")
