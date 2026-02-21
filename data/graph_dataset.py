"""PyTorch Dataset for graph-structured traffic time-series.

Produces sliding-window samples of (input, target) pairs over a
node-time matrix for spatio-temporal forecasting.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple


class GraphTimeSeriesDataset(Dataset):
    """Dataset for graph-based traffic forecasting.

    Takes a (T, N, F) array and produces sliding-window samples:
        input:  (N_local, input_steps, F)
        target: (N_local, forecast_steps)

    N_local is the number of nodes in this client's subgraph (could be
    the full graph for centralized training).

    Args:
        data: np.ndarray of shape (T, N, F) — full time-series
        adjacency: np.ndarray of shape (N, N) — adjacency matrix
        input_steps: Number of past time steps as input
        forecast_steps: Number of future time steps to predict
        stride: Sliding window stride (default 1 = every possible window)
        node_indices: Optional subset of node indices for subgraph training.
                      If None, uses all nodes.
    """

    def __init__(self, data: np.ndarray, adjacency: np.ndarray,
                 input_steps: int = 3, forecast_steps: int = 3,
                 stride: int = 1,
                 node_indices: Optional[np.ndarray] = None):
        super().__init__()

        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.window = input_steps + forecast_steps

        # Select subgraph nodes if specified
        if node_indices is not None:
            self.data = data[:, node_indices, :]
            self.adjacency = adjacency[np.ix_(node_indices, node_indices)]
            self.node_indices = node_indices
        else:
            self.data = data
            self.adjacency = adjacency
            self.node_indices = np.arange(data.shape[1])

        self.num_nodes = self.data.shape[1]
        self.num_features = self.data.shape[2]

        # Compute valid window start positions
        T = self.data.shape[0]
        self.starts = list(range(0, T - self.window + 1, stride))

        # Pre-convert adjacency to tensor
        self.adj_tensor = torch.FloatTensor(self.adjacency)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one sliding-window sample.

        Returns:
            x: (N, input_steps, F) input time-series
            y: (N, forecast_steps) target values (first feature only)
        """
        t = self.starts[idx]

        # Input: (N, T_in, F)
        x = self.data[t:t + self.input_steps]         # (T_in, N, F)
        x = np.transpose(x, (1, 0, 2))                 # (N, T_in, F)

        # Target: (N, T_out) — predict first feature (congestion score)
        y = self.data[t + self.input_steps:t + self.window]  # (T_out, N, F)
        y = y[:, :, 0].T                                      # (N, T_out)

        return torch.FloatTensor(x), torch.FloatTensor(y)

    def get_adjacency(self) -> torch.Tensor:
        """Return the adjacency matrix as a tensor."""
        return self.adj_tensor

    def get_num_nodes(self) -> int:
        """Return number of nodes in this dataset's subgraph."""
        return self.num_nodes


def graph_collate_fn(batch):
    """Custom collate for graph time-series batches.

    Args:
        batch: List of (x, y) tuples from GraphTimeSeriesDataset

    Returns:
        x: (B, N, T_in, F)
        y: (B, N, T_out)
    """
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
