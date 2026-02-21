"""Graph-aware FL client for subgraph-based federated training.

Each GraphClient owns a subset of cameras (nodes) and trains on its
local subgraph only. No access to other clients' data. Only model
weights are shared via FedAvg — never features or raw data.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

from federated.client import FLClient
from data.graph_dataset import GraphTimeSeriesDataset, graph_collate_fn


class GraphClient(FLClient):
    """Federated learning client for graph-based traffic forecasting.

    Trains a spatio-temporal model (T-GCN or AGCRN) on its local subgraph.
    The adjacency matrix is local — only edges between this client's
    cameras are visible.

    Args:
        client_id: Unique identifier
        model: Graph forecasting model (TGCN or AGCRN)
        dataset: GraphTimeSeriesDataset for this client's subgraph
        adjacency: Local subgraph adjacency matrix (torch.Tensor)
        device: Training device
        batch_size: Local batch size
        lr: Learning rate
    """

    def __init__(self, client_id: str, model: nn.Module,
                 dataset: GraphTimeSeriesDataset,
                 adjacency: torch.Tensor,
                 device: str = 'cpu', batch_size: int = 32,
                 lr: float = 0.001):
        # Don't call super().__init__ directly since we need custom dataloader
        self.client_id = client_id
        self.model = model.__class__(
            in_features=model.in_features,
            hidden_dim=model.hidden_dim,
            forecast_steps=model.forecast_steps,
            num_nodes=dataset.get_num_nodes(),
            **self._get_extra_model_args(model)
        ).to(device)
        # Copy initial weights where possible
        self._load_compatible_weights(model)

        self.dataset = dataset
        self.adjacency = adjacency.to(device)
        self.device = device
        self.batch_size = batch_size
        self.lr = lr

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(dataset) > batch_size,
            collate_fn=graph_collate_fn,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []

    def _get_extra_model_args(self, model: nn.Module) -> Dict:
        """Extract model-specific constructor args beyond the common ones."""
        args = {}
        # AGCRN-specific
        if hasattr(model, 'num_layers'):
            args['num_layers'] = model.num_layers
        if hasattr(model, 'cells') and hasattr(model.cells[0], 'gcn_z'):
            # AGCRN has embed_dim in AVWGCN
            avwgcn = model.cells[0].gcn_z
            if hasattr(avwgcn, 'embed_dim'):
                args['embed_dim'] = avwgcn.embed_dim
        # TGCN dropout
        if hasattr(model, 'output_fc'):
            for layer in model.output_fc:
                if isinstance(layer, nn.Dropout):
                    args['dropout'] = layer.p
                    break
        return args

    def _load_compatible_weights(self, source_model: nn.Module):
        """Load weights that are shape-compatible from source model."""
        source_state = source_model.state_dict()
        target_state = self.model.state_dict()

        compatible = {}
        for name, param in source_state.items():
            if name in target_state and param.shape == target_state[name].shape:
                compatible[name] = param

        if compatible:
            target_state.update(compatible)
            self.model.load_state_dict(target_state)

    def get_num_samples(self) -> int:
        """Return number of local training samples."""
        return len(self.dataset)

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Return current model parameters."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_params(self, params: Dict[str, torch.Tensor]):
        """Update model with parameters from server.

        Only loads weights that are shape-compatible, since subgraph
        models may have different node counts (affecting node-specific
        parameters like AGCRN embeddings).
        """
        current_state = self.model.state_dict()
        for name, param in params.items():
            if name in current_state and param.shape == current_state[name].shape:
                current_state[name] = param
        self.model.load_state_dict(current_state)

    def train_local(self, epochs: int) -> Dict[str, torch.Tensor]:
        """Train on local subgraph data.

        Args:
            epochs: Number of local epochs

        Returns:
            Updated model state_dict
        """
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for x_batch, y_batch in self.dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass with local adjacency
                predictions = self._forward(x_batch)
                loss = self.criterion(predictions, y_batch)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.train_losses.append(avg_loss)

        return self.get_model_params()

    def _forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph model with local adjacency.

        Args:
            data: (B, N_local, T_in, F) input time-series

        Returns:
            (B, N_local, T_out) predictions
        """
        # Check if model accepts adj (TGCN requires it, AGCRN optional)
        if hasattr(self.model, 'tgcn_cell'):
            # TGCN — adjacency required
            return self.model.forward_predict(data, self.adjacency)
        else:
            # AGCRN — adjacency optional
            return self.model.forward_predict(data, self.adjacency)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on local dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in self.dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self._forward(x_batch)
                loss = self.criterion(output, y_batch)

                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)

        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'num_samples': total_samples,
            'num_nodes': self.dataset.get_num_nodes(),
        }
