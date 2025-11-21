# src/decen_learn/attacks/attack.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class AttackResult:
    """Output from attack crafting."""
    malicious_weights: Dict[str, np.ndarray]
    metadata: dict = None


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
        """
        Craft malicious weights based on observed honest updates.
        
        Args:
            honest_weights: List of weight dicts from honest neighbors
            attacker_weights: The attacker's current weights
            
        Returns:
            Malicious weight dict to broadcast
        """
        pass