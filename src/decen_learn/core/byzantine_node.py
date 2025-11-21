# src/decen_learn/core/byzantine_node.py
class ByzantineNode(Node):
    """A malicious node that crafts adversarial updates."""
    
    def __init__(
        self,
        node_id: int,
        model: torch.nn.Module,
        projector: WeightProjector,
        config: "NodeConfig",
        attack: "BaseAttack",
        device: Optional[torch.device] = None,
    ):
        super().__init__(node_id, model, projector, config, device)
        self._attack = attack
    
    @property
    def is_byzantine(self) -> bool:
        return True
    
    def prepare_broadcast(self) -> Dict[str, np.ndarray]:
        """Craft malicious weights to send."""
        # Collect honest neighbors' weights for attack computation
        honest_weights = [
            w for w in self.state.buffer 
            if not self._is_byzantine_weight(w)
        ]
        
        if not honest_weights:
            return self.state.weights
        
        return self._attack.craft(honest_weights, self.state.weights)
    
    def update_weights(self, new_weights: Dict[str, np.ndarray], momentum: float = 0.0) -> None:
        """Byzantine nodes ignore consensus updates."""
        pass