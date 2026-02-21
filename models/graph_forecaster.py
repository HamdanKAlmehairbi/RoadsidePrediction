"""T-GCN: Temporal Graph Convolutional Network for traffic forecasting.

Implements GCN (matrix-multiply based, no torch-geometric dependency) inside
GRU gates for spatio-temporal graph forecasting.

Reference: Zhao et al., "T-GCN: A Temporal Graph Convolutional Network
for Traffic Prediction", IEEE TITS 2020.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GraphConvolution(nn.Module):
    """Single graph convolution layer: sigma(A_hat @ X @ W).

    No torch-geometric dependency — pure matrix multiply.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, F_in) node features
            adj: (N, N) adjacency matrix (already normalized)

        Returns:
            (B, N, F_out) transformed features
        """
        # x @ W -> (B, N, F_out)
        support = torch.matmul(x, self.weight)
        # A @ (X @ W) -> (B, N, F_out)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """Compute symmetric normalized adjacency: D^{-1/2} A D^{-1/2}.

    Args:
        adj: (N, N) adjacency matrix
        add_self_loops: Whether to add self-loops (A + I)

    Returns:
        (N, N) normalized adjacency
    """
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    # Degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # D^{-1/2} A D^{-1/2}
    d_mat = torch.diag(deg_inv_sqrt)
    return torch.mm(torch.mm(d_mat, adj), d_mat)


class TGCNCell(nn.Module):
    """T-GCN cell: GCN inside GRU gates.

    Replaces linear transforms in GRU with graph convolutions,
    enabling spatial information flow through the graph structure.

    Args:
        in_features: Input feature dimension per node
        hidden_dim: Hidden state dimension per node
    """

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRU gates with GCN (input + hidden concatenated)
        # Update gate z
        self.gcn_z = GraphConvolution(in_features + hidden_dim, hidden_dim)
        # Reset gate r
        self.gcn_r = GraphConvolution(in_features + hidden_dim, hidden_dim)
        # Candidate hidden state
        self.gcn_h = GraphConvolution(in_features + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """Forward one time step.

        Args:
            x: (B, N, F) input features at current time step
            h: (B, N, hidden_dim) previous hidden state
            adj: (N, N) normalized adjacency matrix

        Returns:
            (B, N, hidden_dim) new hidden state
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)  # (B, N, F + hidden_dim)

        # Update gate
        z = torch.sigmoid(self.gcn_z(combined, adj))
        # Reset gate
        r = torch.sigmoid(self.gcn_r(combined, adj))

        # Candidate hidden state
        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.gcn_h(combined_r, adj))

        # New hidden state
        h_new = z * h + (1 - z) * h_tilde
        return h_new


class TGCN(nn.Module):
    """Temporal Graph Convolutional Network for traffic forecasting.

    Processes a sequence of graph-structured time-series and forecasts
    future values at each node.

    Args:
        in_features: Number of input features per node per time step
        hidden_dim: Hidden dimension of GRU cells
        forecast_steps: Number of future steps to predict
        num_nodes: Number of nodes (parameterized for subgraph training)
        dropout: Dropout rate
    """

    def __init__(self, in_features: int = 1, hidden_dim: int = 64,
                 forecast_steps: int = 12, num_nodes: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps
        self.num_nodes = num_nodes

        # T-GCN recurrent cell
        self.tgcn_cell = TGCNCell(in_features, hidden_dim)

        # Output projection: hidden state -> forecast
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass — raw output (unbounded).

        Args:
            x: (B, N, T_in, F) input time-series
            adj: (N, N) adjacency matrix (raw, will be normalized internally)

        Returns:
            (B, N, T_out) raw forecast scores
        """
        B, N, T, F = x.shape

        # Normalize adjacency
        adj_norm = normalize_adjacency(adj)

        # Initialize hidden state
        h = torch.zeros(B, N, self.hidden_dim, device=x.device)

        # Process each time step
        for t in range(T):
            x_t = x[:, :, t, :]  # (B, N, F)
            h = self.tgcn_cell(x_t, h, adj_norm)

        # Project hidden state to forecast
        # h: (B, N, hidden_dim) -> (B, N, forecast_steps)
        output = self.output_fc(h)
        return output

    def forward_predict(self, x: torch.Tensor,
                        adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid bounding to [0, 1].

        Args:
            x: (B, N, T_in, F) input time-series
            adj: (N, N) adjacency matrix

        Returns:
            (B, N, T_out) forecast scores in [0, 1]
        """
        return torch.sigmoid(self.forward(x, adj))

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
