"""Centralized training for multimodal congestion prediction (baseline)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from training.metrics import MetricTracker, compute_metrics


class CentralizedTrainer:
    """Centralized trainer for multimodal congestion prediction.

    Serves as a baseline to compare against federated training.

    Args:
        model: FusionModel or single-modality model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        lr: Learning rate
        weight_decay: L2 regularization
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cpu', lr: float = 0.001,
                 weight_decay: float = 0.0001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

        # Training history
        self.history = {
            'train_loss': [],
            'train_mse': [],
            'train_mae': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_r2': []
        }

    def train_epoch(self, multimodal: bool = True) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            multimodal: Whether data is multimodal (camera + sensor)

        Returns:
            Dict with training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            if multimodal:
                camera_data, sensor_data, targets = batch
                camera_data = camera_data.to(self.device)
                sensor_data = sensor_data.to(self.device)
            else:
                data, targets = batch
                data = data.to(self.device)

            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if multimodal:
                outputs = self.model(camera_data, sensor_data)
            else:
                outputs = self.model.forward_predict(data)

            outputs = outputs.squeeze()
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            self.train_metrics.update(outputs, targets)

        metrics = self.train_metrics.compute()
        metrics['loss'] = total_loss / num_batches

        return metrics

    def validate(self, multimodal: bool = True) -> Dict[str, float]:
        """Validate model.

        Args:
            multimodal: Whether data is multimodal

        Returns:
            Dict with validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if multimodal:
                    camera_data, sensor_data, targets = batch
                    camera_data = camera_data.to(self.device)
                    sensor_data = sensor_data.to(self.device)
                else:
                    data, targets = batch
                    data = data.to(self.device)

                targets = targets.to(self.device)

                if multimodal:
                    outputs = self.model(camera_data, sensor_data)
                else:
                    outputs = self.model.forward_predict(data)

                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                self.val_metrics.update(outputs, targets)

        metrics = self.val_metrics.compute()
        metrics['loss'] = total_loss / num_batches

        return metrics

    def train(self, epochs: int, multimodal: bool = True,
              early_stopping_patience: int = 10,
              save_best: Optional[str] = None) -> Dict:
        """Full training loop.

        Args:
            epochs: Number of training epochs
            multimodal: Whether data is multimodal
            early_stopping_patience: Epochs without improvement before stopping
            save_best: Path to save best model

        Returns:
            Dict with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(multimodal)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_mae'].append(train_metrics['mae'])

            # Validate
            val_metrics = self.validate(multimodal)
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_mse'].append(val_metrics['mse'])
                self.history['val_mae'].append(val_metrics['mae'])
                self.history['val_r2'].append(val_metrics['r2'])

                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    if save_best:
                        self.save_checkpoint(save_best, epoch)
                else:
                    patience_counter += 1

            # Logging
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        return self.history

    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint.

        Args:
            path: Save path
            epoch: Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint.

        Args:
            path: Checkpoint path

        Returns:
            Epoch number
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']
