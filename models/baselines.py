"""Baseline models for traffic congestion forecasting comparison.

All baselines implement the same interface as TGCN/AGCRN:
    forward(x) -> (B, N, T_out) raw scores
    forward_predict(x) -> (B, N, T_out) bounded [0, 1]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class HistoricalAverage(nn.Module):
    """Historical Average baseline.

    Predicts future congestion as the mean of all observed values for that
    node. At test time, simply repeats the mean of the input window.

    This is a non-parametric baseline (no trainable weights).
    """

    def __init__(self, forecast_steps: int = 12):
        super().__init__()
        self.forecast_steps = forecast_steps

    def forward(self, x: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict mean of input for each node.

        Args:
            x: (B, N, T_in, F) input time-series
            adj: Unused, for interface compatibility

        Returns:
            (B, N, T_out) repeated mean values
        """
        # Mean over time and features -> (B, N)
        mean_val = x.mean(dim=(2, 3))
        # Repeat for each forecast step -> (B, N, T_out)
        return mean_val.unsqueeze(-1).expand(-1, -1, self.forecast_steps)

    def forward_predict(self, x: torch.Tensor,
                        adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict with clipping to [0, 1]."""
        return self.forward(x, adj).clamp(0, 1)

    def get_num_params(self) -> int:
        return 0


class PersistenceModel(nn.Module):
    """Persistence (last-value) baseline.

    Predicts that future congestion = current congestion.
    "Congestion in 1 hour = congestion now."

    This is a non-parametric baseline (no trainable weights).
    """

    def __init__(self, forecast_steps: int = 12):
        super().__init__()
        self.forecast_steps = forecast_steps

    def forward(self, x: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict last observed value for each node.

        Args:
            x: (B, N, T_in, F) input time-series
            adj: Unused, for interface compatibility

        Returns:
            (B, N, T_out) repeated last values
        """
        # Last time step, mean over features -> (B, N)
        last_val = x[:, :, -1, :].mean(dim=-1)
        # Repeat for each forecast step -> (B, N, T_out)
        return last_val.unsqueeze(-1).expand(-1, -1, self.forecast_steps)

    def forward_predict(self, x: torch.Tensor,
                        adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict with clipping to [0, 1]."""
        return self.forward(x, adj).clamp(0, 1)

    def get_num_params(self) -> int:
        return 0


class PerNodeLSTM(nn.Module):
    """Independent LSTM per node (no graph structure).

    Each node has its own LSTM — tests whether graph structure adds value
    beyond per-node temporal modeling.

    Args:
        input_dim: Input features per node per time step
        hidden_dim: LSTM hidden dimension
        forecast_steps: Number of future steps to predict
        num_nodes: Number of nodes (each gets its own LSTM head)
        dropout: Dropout rate
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32,
                 forecast_steps: int = 12, num_nodes: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps
        self.num_nodes = num_nodes

        # Shared LSTM across nodes (parameter-efficient)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Per-node output head
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, forecast_steps),
        )

    def forward(self, x: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass — processes each node independently.

        Args:
            x: (B, N, T_in, F) input time-series
            adj: Unused, for interface compatibility

        Returns:
            (B, N, T_out) raw forecast scores
        """
        B, N, T, F = x.shape

        # Reshape to process all nodes through shared LSTM
        # (B*N, T, F)
        x_flat = x.reshape(B * N, T, F)

        # LSTM forward
        lstm_out, _ = self.lstm(x_flat)  # (B*N, T, hidden_dim)

        # Use last hidden state
        h_last = lstm_out[:, -1, :]  # (B*N, hidden_dim)

        # Output projection
        output = self.output_fc(h_last)  # (B*N, forecast_steps)

        # Reshape back
        return output.reshape(B, N, self.forecast_steps)

    def forward_predict(self, x: torch.Tensor,
                        adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with sigmoid bounding."""
        return torch.sigmoid(self.forward(x, adj))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VARModel(nn.Module):
    """Vector AutoRegression baseline (linear multivariate).

    Simple linear model: y_t = W @ [x_{t-1}, ..., x_{t-p}] + b
    Tests whether deep learning provides benefit over classical methods.

    Args:
        num_nodes: Number of nodes (variables)
        lags: Number of lag steps to use
        forecast_steps: Number of future steps to predict
        input_dim: Features per node per step
    """

    def __init__(self, num_nodes: int = 50, lags: int = 12,
                 forecast_steps: int = 12, input_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.lags = lags
        self.forecast_steps = forecast_steps

        # Linear transform: flatten all nodes x lags x features -> forecast
        flat_input = num_nodes * lags * input_dim
        flat_output = num_nodes * forecast_steps

        self.linear = nn.Linear(flat_input, flat_output)

    def forward(self, x: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, T_in, F) input time-series
            adj: Unused, for interface compatibility

        Returns:
            (B, N, T_out) raw forecast scores
        """
        B, N, T, F = x.shape

        # Use last `lags` time steps
        if T > self.lags:
            x = x[:, :, -self.lags:, :]

        # Flatten: (B, N * lags * F)
        x_flat = x.reshape(B, -1)

        # Pad if input is smaller than expected
        expected = self.linear.in_features
        if x_flat.size(1) < expected:
            pad = torch.zeros(B, expected - x_flat.size(1), device=x.device)
            x_flat = torch.cat([x_flat, pad], dim=1)
        elif x_flat.size(1) > expected:
            x_flat = x_flat[:, :expected]

        # Linear transform
        output = self.linear(x_flat)  # (B, N * T_out)
        return output.reshape(B, self.num_nodes, self.forecast_steps)

    def forward_predict(self, x: torch.Tensor,
                        adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward with sigmoid bounding."""
        return torch.sigmoid(self.forward(x, adj))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
