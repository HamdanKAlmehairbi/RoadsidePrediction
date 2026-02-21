"""Camera-specific FL client for roadside camera nodes."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict

from .client import FLClient


class CameraClient(FLClient):
    """Federated learning client for roadside camera nodes.

    Handles image-based congestion prediction training locally.

    Args:
        client_id: Unique identifier
        model: CameraCNN model
        dataset: CameraDataset instance
        device: Training device
        batch_size: Local batch size
        lr: Learning rate
    """

    def __init__(self, client_id: str, model: nn.Module, dataset: Dataset,
                 device: str = 'cpu', batch_size: int = 32, lr: float = 0.001):
        super().__init__(client_id, model, dataset, device, batch_size, lr)

    def train_local(self, epochs: int) -> Dict[str, torch.Tensor]:
        """Train on local camera data.

        Args:
            epochs: Number of local epochs

        Returns:
            Updated model state_dict
        """
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model.forward_predict(images)
                loss = self.criterion(outputs.squeeze(), targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.train_losses.append(avg_loss)

        return self.get_model_params()

    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass for camera data."""
        return self.model.forward_predict(data)

    def get_feature_extractor_params(self) -> Dict[str, torch.Tensor]:
        """Get only the feature extractor parameters (for partial aggregation).

        Useful when only aggregating shared feature extractors in FL.

        Returns:
            State dict of feature extraction layers only
        """
        params = {}
        for name, param in self.model.named_parameters():
            if 'fc_predict' not in name:  # Exclude prediction head
                params[name] = param.cpu().clone()
        return params

    def set_feature_extractor_params(self, params: Dict[str, torch.Tensor]):
        """Set only the feature extractor parameters.

        Args:
            params: State dict for feature extraction layers
        """
        current_state = self.model.state_dict()
        for name, param in params.items():
            if name in current_state:
                current_state[name] = param
        self.model.load_state_dict(current_state)
