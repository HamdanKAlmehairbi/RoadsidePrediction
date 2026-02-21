"""Federated training orchestration for multimodal congestion prediction."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from federated.server import FLServer
from federated.client import FLClient
from training.metrics import compute_metrics


class FederatedTrainer:
    """Orchestrates federated training across camera and vehicle clients.

    Manages the full federated learning process including client selection,
    local training, aggregation, and evaluation.

    Args:
        server: FLServer instance with global model
        camera_clients: List of camera FL clients
        vehicle_clients: List of vehicle FL clients
        test_loader: Optional test data loader for evaluation
        device: Device for evaluation
    """

    def __init__(self, server: FLServer,
                 camera_clients: List[FLClient],
                 vehicle_clients: List[FLClient],
                 test_loader: Optional[DataLoader] = None,
                 device: str = 'cpu'):
        self.server = server
        self.camera_clients = camera_clients
        self.vehicle_clients = vehicle_clients
        self.all_clients = camera_clients + vehicle_clients
        self.test_loader = test_loader
        self.device = device

        # Training history
        self.history = {
            'round': [],
            'camera_clients_trained': [],
            'vehicle_clients_trained': [],
            'total_samples': [],
            'test_mse': [],
            'test_mae': [],
            'test_r2': []
        }

    def train(self, rounds: int, local_epochs: int,
              camera_fraction: float = 1.0,
              vehicle_fraction: float = 1.0,
              evaluate_every: int = 1,
              save_every: int = 10,
              checkpoint_dir: Optional[str] = None) -> Dict:
        """Run federated training.

        Args:
            rounds: Number of FL communication rounds
            local_epochs: Local training epochs per round
            camera_fraction: Fraction of camera clients to sample
            vehicle_fraction: Fraction of vehicle clients to sample
            evaluate_every: Evaluate global model every N rounds
            save_every: Save checkpoint every N rounds
            checkpoint_dir: Directory for checkpoints

        Returns:
            Training history dict
        """
        for round_num in tqdm(range(rounds), desc="FL Rounds"):
            # Select clients
            selected_camera = self._sample_clients(
                self.camera_clients, camera_fraction
            )
            selected_vehicle = self._sample_clients(
                self.vehicle_clients, vehicle_fraction
            )

            # Broadcast global model to selected clients
            global_params = self.server.broadcast()

            for client in selected_camera + selected_vehicle:
                client.set_model_params(global_params)

            # Local training
            all_updates = []
            all_weights = []

            # Train camera clients
            for client in selected_camera:
                updates = client.train_local(local_epochs)
                all_updates.append(updates)
                all_weights.append(client.get_num_samples())

            # Train vehicle clients
            for client in selected_vehicle:
                updates = client.train_local(local_epochs)
                all_updates.append(updates)
                all_weights.append(client.get_num_samples())

            # Aggregate at server
            self.server.aggregate(all_updates, all_weights)

            # Log round stats
            self.history['round'].append(round_num + 1)
            self.history['camera_clients_trained'].append(len(selected_camera))
            self.history['vehicle_clients_trained'].append(len(selected_vehicle))
            self.history['total_samples'].append(sum(all_weights))

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

            # Save checkpoint
            if checkpoint_dir and (round_num + 1) % save_every == 0:
                path = f"{checkpoint_dir}/round_{round_num + 1}.pt"
                self.server.save_checkpoint(path, round_num + 1)

        return self.history

    def _sample_clients(self, clients: List[FLClient],
                        fraction: float) -> List[FLClient]:
        """Sample clients for a round.

        Args:
            clients: List of clients to sample from
            fraction: Fraction to sample

        Returns:
            Selected clients
        """
        if fraction >= 1.0 or len(clients) == 0:
            return clients

        num_selected = max(1, int(len(clients) * fraction))
        indices = torch.randperm(len(clients))[:num_selected].tolist()
        return [clients[i] for i in indices]

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate global model on test set.

        Returns:
            Dict with test metrics
        """
        return self.server.evaluate_global(self.test_loader)

    def train_single_modality(self, modality: str, rounds: int,
                              local_epochs: int,
                              client_fraction: float = 1.0) -> Dict:
        """Train using only one modality (baseline).

        Args:
            modality: "camera" or "vehicle"
            rounds: Number of FL rounds
            local_epochs: Local epochs per round
            client_fraction: Client sampling fraction

        Returns:
            Training history
        """
        if modality == "camera":
            clients = self.camera_clients
        elif modality == "vehicle":
            clients = self.vehicle_clients
        else:
            raise ValueError(f"Unknown modality: {modality}")

        history = {
            'round': [],
            'clients_trained': [],
            'total_samples': [],
            'test_mse': [],
            'test_mae': [],
            'test_r2': []
        }

        for round_num in tqdm(range(rounds), desc=f"FL Rounds ({modality})"):
            selected = self._sample_clients(clients, client_fraction)

            # Broadcast and train
            global_params = self.server.broadcast()
            for client in selected:
                client.set_model_params(global_params)

            updates = []
            weights = []
            for client in selected:
                update = client.train_local(local_epochs)
                updates.append(update)
                weights.append(client.get_num_samples())

            self.server.aggregate(updates, weights)

            # Log
            history['round'].append(round_num + 1)
            history['clients_trained'].append(len(selected))
            history['total_samples'].append(sum(weights))

            if self.test_loader:
                metrics = self._evaluate()
                history['test_mse'].append(metrics['mse'])
                history['test_mae'].append(metrics['mae'])
                history['test_r2'].append(metrics['r2'])

        return history

    def get_client_statistics(self) -> Dict:
        """Get statistics about clients and data distribution.

        Returns:
            Dict with client statistics
        """
        camera_samples = [c.get_num_samples() for c in self.camera_clients]
        vehicle_samples = [c.get_num_samples() for c in self.vehicle_clients]

        return {
            'num_camera_clients': len(self.camera_clients),
            'num_vehicle_clients': len(self.vehicle_clients),
            'total_camera_samples': sum(camera_samples),
            'total_vehicle_samples': sum(vehicle_samples),
            'avg_camera_samples': sum(camera_samples) / len(camera_samples) if camera_samples else 0,
            'avg_vehicle_samples': sum(vehicle_samples) / len(vehicle_samples) if vehicle_samples else 0,
            'camera_sample_distribution': camera_samples,
            'vehicle_sample_distribution': vehicle_samples
        }
