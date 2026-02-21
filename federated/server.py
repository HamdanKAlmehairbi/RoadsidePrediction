"""FL server implementation for coordinating federated training."""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import copy

from .client import FLClient
from .fedavg import fedavg_aggregate


class FLServer:
    """Federated learning server for coordinating distributed training.

    Manages global model, aggregates client updates, and broadcasts
    updated parameters.

    Args:
        global_model: Initial global model
        aggregation_strategy: Aggregation method ("fedavg", "weighted_avg")
        device: Device for server-side computation
    """

    def __init__(self, global_model: nn.Module,
                 aggregation_strategy: str = "fedavg",
                 device: str = 'cpu'):
        self.global_model = copy.deepcopy(global_model).to(device)
        self.aggregation_strategy = aggregation_strategy
        self.device = device

        # Training history
        self.round_metrics = []

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters.

        Returns:
            State dict of global model
        """
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def broadcast(self) -> Dict[str, torch.Tensor]:
        """Broadcast global model parameters to clients.

        Returns:
            State dict for clients to load
        """
        return self.get_global_params()

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]],
                  weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates.

        Args:
            client_updates: List of state_dicts from clients
            weights: Optional weights (e.g., num_samples per client)

        Returns:
            Aggregated state_dict
        """
        if len(client_updates) == 0:
            return self.get_global_params()

        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0] * len(client_updates)

        if self.aggregation_strategy == "fedavg":
            aggregated = fedavg_aggregate(client_updates, weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

        # Update global model
        self.global_model.load_state_dict(aggregated)

        return aggregated

    def aggregate_round(self, clients: List[FLClient],
                       local_epochs: int,
                       client_fraction: float = 1.0) -> Dict[str, float]:
        """Execute one round of federated training.

        Args:
            clients: List of FL clients
            local_epochs: Number of local training epochs
            client_fraction: Fraction of clients to sample (default: all)

        Returns:
            Dict with round metrics
        """
        # Sample clients
        num_selected = max(1, int(len(clients) * client_fraction))
        selected_indices = torch.randperm(len(clients))[:num_selected].tolist()
        selected_clients = [clients[i] for i in selected_indices]

        # Broadcast global model to selected clients
        global_params = self.broadcast()
        for client in selected_clients:
            client.set_model_params(global_params)

        # Local training
        client_updates = []
        client_weights = []

        for client in selected_clients:
            # Train locally
            updated_params = client.train_local(local_epochs)
            client_updates.append(updated_params)
            client_weights.append(client.get_num_samples())

        # Aggregate updates
        self.aggregate(client_updates, client_weights)

        # Compute metrics
        metrics = {
            'num_clients': len(selected_clients),
            'total_samples': sum(client_weights),
            'avg_samples_per_client': sum(client_weights) / len(selected_clients)
        }
        self.round_metrics.append(metrics)

        return metrics

    def evaluate_global(self, test_loader, criterion=None) -> Dict[str, float]:
        """Evaluate global model on test data.

        Args:
            test_loader: DataLoader with test data
            criterion: Loss function (default: MSELoss)

        Returns:
            Dict with evaluation metrics
        """
        if criterion is None:
            criterion = nn.MSELoss()

        self.global_model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                # Handle both single and multimodal data
                if isinstance(data, (list, tuple)):
                    data = [d.to(self.device) for d in data]
                    output = self.global_model(*data)
                else:
                    data = data.to(self.device)
                    output = self.global_model.forward_predict(data)

                target = target.to(self.device)
                loss = criterion(output.squeeze(), target)

                total_loss += loss.item() * target.size(0)
                total_samples += target.size(0)

                all_preds.extend(output.squeeze().cpu().tolist())
                all_targets.extend(target.cpu().tolist())

        # Compute metrics
        mse = total_loss / total_samples
        mae = sum(abs(p - t) for p, t in zip(all_preds, all_targets)) / total_samples

        # R-squared
        mean_target = sum(all_targets) / len(all_targets)
        ss_tot = sum((t - mean_target) ** 2 for t in all_targets)
        ss_res = sum((t - p) ** 2 for t, p in zip(all_targets, all_preds))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'num_samples': total_samples
        }

    def evaluate_global_graph(self, test_loader, adjacency,
                              criterion=None) -> Dict[str, float]:
        """Evaluate global graph model on test data.

        Handles adjacency passing for graph models (TGCN/AGCRN).

        Args:
            test_loader: DataLoader with (x, y) graph time-series batches
            adjacency: Adjacency matrix tensor for the test graph
            criterion: Loss function (default: MSELoss)

        Returns:
            Dict with mse, mae, r2, num_samples
        """
        if criterion is None:
            criterion = nn.MSELoss()

        self.global_model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                adj = adjacency.to(self.device)

                output = self.global_model.forward_predict(x_batch, adj)
                loss = criterion(output, y_batch)

                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)

                all_preds.append(output.cpu())
                all_targets.append(y_batch.cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        mse = total_loss / total_samples if total_samples > 0 else 0.0
        mae = torch.mean(torch.abs(preds - targets)).item()

        # R-squared
        ss_res = torch.sum((targets - preds) ** 2).item()
        ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'num_samples': total_samples,
        }

    def save_checkpoint(self, path: str, round_num: int):
        """Save server checkpoint.

        Args:
            path: Path to save checkpoint
            round_num: Current round number
        """
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.global_model.state_dict(),
            'round_metrics': self.round_metrics
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """Load server checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Round number from checkpoint
        """
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.round_metrics = checkpoint.get('round_metrics', [])
        return checkpoint['round']
