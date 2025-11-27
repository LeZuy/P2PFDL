import torch
from typing import Dict, List
from .base import BaseAttack

class GlobalIPMAttack(BaseAttack):
    """
    IPM-global attack: Byzantine nodes compute attack using 
    the GLOBAL honest mean, not neighborhood mean.

    This matches the IPM definition used in the DFL paper (2024):
        theta_mal = -eps * mean(honest_global_models)

    All Byzantine nodes will send the same malicious update.
    """

    def __init__(self, eps: float = 0.5):
        super().__init__()
        self.eps = eps

    def craft_global(
        self,
        global_honest_weights: List[Dict[str, torch.Tensor]],
        attacker_template: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Craft malicious update using GLOBAL honest models.

        Args:
            global_honest_weights: list of weight dicts from ALL honest clients
            attacker_template: structure for unflattening

        Returns:
            Malicious weight dict
        """

        if not global_honest_weights:
            return self._clone_template(attacker_template)

        # Flatten honest models
        flat = torch.stack(
            [self._flatten(w) for w in global_honest_weights],
            dim=0
        )

        # Compute GLOBAL honest mean
        honest_mean = flat.mean(dim=0)

        # IPM global malicious update
        malicious_flat = -self.eps * honest_mean

        # Restore dictionary structure
        return self._unflatten(malicious_flat, attacker_template)

    # attacker must use craft_global() instead of normal craft()
    def craft(self, honest_weights, attacker_weights):
        raise RuntimeError(
            "GlobalIPMAttack must be invoked through craft_global()"
        )
