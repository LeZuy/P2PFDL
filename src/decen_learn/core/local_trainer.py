"""Local training management for decentralized nodes."""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class LocalTrainer:
    """Manages local SGD training for a node.
    
    Handles:
    - Optimizer and scheduler management
    - Training loop execution
    - Metric tracking
    """
    
    def __init__(
        self,
        node_id: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        max_epochs: int = 200,
    ):
        """Initialize local trainer.
        
        Args:
            model: PyTorch model to train
            dataloader: Training data loader
            device: Device for training
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: L2 regularization
            max_epochs: For scheduler
        """
        self.node_id = node_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.model._optimizer = self.optimizer

        self.model.to(self.device)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs
        )
        
        self._current_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        # self.model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            try:
                # print(f"[Node {self.node_id}] trainer.device = {self.device}")
                # print(f"[Node {self.node_id}] inputs.device = {inputs.device}")
                # first_param = next(self.model.parameters())
                # print(f"[Node {self.node_id}] model param device = {first_param.device}")

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            except RuntimeError as e:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    stats = {
                        "allocated": torch.cuda.memory_allocated(self.device),
                        "reserved": torch.cuda.memory_reserved(self.device),
                        "max_allocated": torch.cuda.max_memory_allocated(self.device),
                        "max_reserved": torch.cuda.max_memory_reserved(self.device),
                    }
                    print(f"[Node {self.node_id}] CUDA runtime error: {e}")
                    print(f"Memory stats: {stats}")
                raise

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # self.scheduler.step()
        self._current_epoch += 1

        avg_loss = total_loss / len(self.dataloader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None
    ) -> Tuple[float, float]:
        """Evaluate model on provided dataloader.
        
        Args:
            dataloader: Test dataloader (uses training data if None)
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        if dataloader is None:
            dataloader = self.dataloader
        
        self.model.eval()
        self.model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_epoch(self) -> int:
        """Get current epoch number."""
        return self._current_epoch