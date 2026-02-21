"""Federated training orchestration for graph-based congestion forecasting.

Each client trains on its own subgraph. Server aggregates via FedAvg.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm

from federated.server import FLServer
from federated.graph_client import GraphClient
from training.metrics import compute_metrics, compute_per_horizon_metrics
from data.graph_dataset import graph_collate_fn


class GraphFederatedTrainer:
    """Orchestrates federated training for graph forecasting models.

    Args:
        server: FLServer with global graph model
        graph_clients: List of GraphClient instances (subgraph training)
        test_loader: Optional test DataLoader (full graph)
        test_adjacency: Adjacency for test evaluation (full graph)
        device: Device for evaluation
    """

    def __init__(self, server: FLServer,
                 graph_clients: List[GraphClient],
                 test_loader: Optional[DataLoader] = None,
                 test_adjacency: Optional[torch.Tensor] = None,
                 device: str = 'cpu'):
        self.server = server
        self.graph_clients = graph_clients
        self.test_loader = test_loader
        self.test_adjacency = test_adjacency.to(device) if test_adjacency is not None else None
        self.device = device

        self.history = {
            'round': [],
            'clients_trained': [],
            'total_samples': [],
            'avg_client_loss': [],
            'test_mse': [],
            'test_mae': [],
            'test_r2': [],
        }

    def train(self, rounds: int, local_epochs: int,
              client_fraction: float = 1.0,
              evaluate_every: int = 1,
              save_every: int = 10,
              checkpoint_dir: Optional[str] = None) -> Dict:
        """Run federated training.

        Args:
            rounds: Number of FL communication rounds
            local_epochs: Local epochs per round
            client_fraction: Fraction of clients to sample per round
            evaluate_every: Evaluate every N rounds
            save_every: Save checkpoint every N rounds
            checkpoint_dir: Directory for checkpoints

        Returns:
            Training history dict
        """
        for round_num in tqdm(range(rounds), desc="FL Rounds"):
            # Select clients
            selected = self._sample_clients(client_fraction)

            # Broadcast global model
            global_params = self.server.broadcast()
            for client in selected:
                client.set_model_params(global_params)

            # Local training
            all_updates = []
            all_weights = []
            round_losses = []

            for client in selected:
                updates = client.train_local(local_epochs)
                all_updates.append(updates)
                all_weights.append(client.get_num_samples())
                if client.train_losses:
                    round_losses.append(client.train_losses[-1])

            # Aggregate
            self.server.aggregate(all_updates, all_weights)

            # Log
            self.history['round'].append(round_num + 1)
            self.history['clients_trained'].append(len(selected))
            self.history['total_samples'].append(sum(all_weights))
            self.history['avg_client_loss'].append(
                sum(round_losses) / len(round_losses) if round_losses else 0.0
            )

            # Evaluate
            if (round_num + 1) % evaluate_every == 0 and self.test_loader:
                metrics = self._evaluate()
                self.history['test_mse'].append(metrics['mse'])
                self.history['test_mae'].append(metrics['mae'])
                self.history['test_r2'].append(metrics['r2'])

                print(f"Round {round_num + 1}/{rounds} - "
                      f"Test MSE: {metrics['mse']:.4f}, "
                      f"MAE: {metrics['mae']:.4f}, "
                      f"R2: {metrics['r2']:.4f}")
            else:
                self.history['test_mse'].append(None)
                self.history['test_mae'].append(None)
                self.history['test_r2'].append(None)

            # Checkpoint
            if checkpoint_dir and (round_num + 1) % save_every == 0:
                path = f"{checkpoint_dir}/round_{round_num + 1}.pt"
                self.server.save_checkpoint(path, round_num + 1)

        return self.history

    def _sample_clients(self, fraction: float) -> List[GraphClient]:
        """Sample clients for a round."""
        if fraction >= 1.0 or len(self.graph_clients) == 0:
            return self.graph_clients

        num_selected = max(1, int(len(self.graph_clients) * fraction))
        indices = torch.randperm(len(self.graph_clients))[:num_selected].tolist()
        return [self.graph_clients[i] for i in indices]

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate global model on test set."""
        if self.test_adjacency is not None:
            return self.server.evaluate_global_graph(
                self.test_loader, self.test_adjacency
            )
        return self.server.evaluate_global(self.test_loader)

    def get_client_statistics(self) -> Dict:
        """Get statistics about client data distribution."""
        samples = [c.get_num_samples() for c in self.graph_clients]
        nodes = [c.dataset.get_num_nodes() for c in self.graph_clients]
        return {
            'num_clients': len(self.graph_clients),
            'total_samples': sum(samples),
            'avg_samples_per_client': sum(samples) / len(samples) if samples else 0,
            'sample_distribution': samples,
            'nodes_per_client': nodes,
        }
