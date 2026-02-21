"""Base FL client implementation."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import copy


class FLClient(ABC):
    """Abstract base class for federated learning clients.

    Provides common functionality for local training and communication
    with the federated server.

    Args:
        client_id: Unique identifier for this client
        model: PyTorch model to train
        dataset: Local dataset for this client
        device: Device to train on ('cpu' or 'cuda')
        batch_size: Local training batch size
        lr: Learning rate for local optimizer
    """

    def __init__(self, client_id: str, model: nn.Module, dataset: Dataset,
                 device: str = 'cpu', batch_size: int = 32, lr: float = 0.001):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.lr = lr

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(dataset) > batch_size
        )
        self.criterion = nn.MSELoss()  # Regression task
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Training history
        self.train_losses = []

    def get_num_samples(self) -> int:
        """Return number of local training samples."""
        return len(self.dataset)

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Return current model parameters."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_params(self, params: Dict[str, torch.Tensor]):
        """Update model with new parameters from server."""
        self.model.load_state_dict(params)

    @abstractmethod
    def train_local(self, epochs: int) -> Dict[str, torch.Tensor]:
        """Train locally and return updated model parameters.

        Args:
            epochs: Number of local training epochs

        Returns:
            Updated model state_dict
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on local dataset.

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self._forward(data)
                loss = self.criterion(output.squeeze(), target)

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'num_samples': total_samples
        }

    @abstractmethod
    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through model (modality-specific).

        Args:
            data: Input data tensor

        Returns:
            Model output
        """
        pass

    def get_gradient_norm(self) -> float:
        """Compute gradient norm for debugging/analysis."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.client_id}, samples={self.get_num_samples()})"
