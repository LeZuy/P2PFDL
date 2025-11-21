# src/decen_learn/attacks/base.py
"""Base attack class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

@dataclass
class AttackResult:
    """Output from attack crafting."""
    malicious_weights: Dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)


class BaseAttack(ABC):
    """Abstract base class for Byzantine attacks."""
    
    def __init__(self, boosting_factor: float = 1.0):
        self.boosting_factor = boosting_factor
    
    @abstractmethod
    def craft(
        self,
        honest_weights: List[Dict[str, np.ndarray]],
        attacker_weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Craft malicious weights.
        
        Args:
            honest_weights: List of weight dicts from honest neighbors
            attacker_weights: The attacker's current weights
            
        Returns:
            Malicious weight dict to broadcast
        """
        pass
    
    def _flatten(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten weight dictionary to 1D vector."""
        return np.concatenate([v.flatten() for v in weights.values()])
    
    def _unflatten(
        self, 
        flat: np.ndarray, 
        template: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Restore flattened vector to weight dictionary using template."""
        result = {}
        pos = 0
        for name, arr in template.items():
            size = arr.size
            result[name] = flat[pos:pos + size].reshape(arr.shape)
            pos += size
        return result