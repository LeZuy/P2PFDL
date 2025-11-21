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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs
        )
        
        self._current_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Execute one training epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        self.model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update learning rate
        self.scheduler.step()
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