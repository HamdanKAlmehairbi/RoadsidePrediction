"""Centralized graph trainer — upper-bound baseline.

Trains on the full graph (all nodes) in a centralized setting.
Used to compare against federated training to measure the FL gap.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm

from data.graph_dataset import graph_collate_fn
from training.metrics import compute_metrics, compute_per_horizon_metrics


class GraphCentralizedTrainer:
    """Centralized trainer for graph-based traffic forecasting.

    Args:
        model: Graph forecasting model (TGCN or AGCRN)
        adjacency: Full graph adjacency matrix (torch.Tensor)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Training device
        lr: Learning rate
    """

    def __init__(self, model: nn.Module, adjacency: torch.Tensor,
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 device: str = 'cpu', lr: float = 0.001,
                 weight_decay: float = 0.0001, scheduler: Optional[str] = None,
                 scheduler_params: Optional[Dict] = None):
        self.model = model.to(device)
        self.adjacency = adjacency.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)

        self.scheduler = None
        if scheduler == 'cosine':
            T_max = (scheduler_params or {}).get('T_max', 50)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-6
            )

    def train(self, epochs: int, early_stopping_patience: int = 10) -> Dict:
        """Train the model.

        Args:
            epochs: Number of training epochs
            early_stopping_patience: Stop if val loss doesn't improve for N epochs

        Returns:
            Training history dict
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
        }
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)

            # Validate
            if self.val_loader is not None:
                val_metrics = self._validate()
                history['val_loss'].append(val_metrics['loss'])
                history['val_mae'].append(val_metrics['mae'])
                history['val_rmse'].append(val_metrics['rmse'])

                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def _train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model.forward_predict(x_batch, self.adjacency)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model.forward_predict(x_batch, self.adjacency)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item()
                num_batches += 1
                all_preds.append(predictions.cpu())
                all_targets.append(y_batch.cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        metrics = compute_metrics(preds.flatten(), targets.flatten())
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

        return metrics

    def evaluate(self, test_loader: DataLoader,
                 step_minutes: int = 5) -> Dict:
        """Full evaluation with per-horizon metrics.

        Args:
            test_loader: DataLoader for test data
            step_minutes: Minutes per time step

        Returns:
            Dict with overall + per-horizon metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model.forward_predict(x_batch, self.adjacency)
                all_preds.append(predictions.cpu())
                all_targets.append(y_batch.cpu())

        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        overall = compute_metrics(preds.flatten(), targets.flatten())
        per_horizon = compute_per_horizon_metrics(preds, targets,
                                                    step_minutes=step_minutes)

        return {
            'overall': overall,
            'per_horizon': per_horizon,
        }
