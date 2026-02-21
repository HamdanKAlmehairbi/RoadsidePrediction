"""AGCRN: Adaptive Graph Convolutional Recurrent Network.

Learns graph structure from data (no predefined adjacency needed) and
uses node-adaptive parameters for heterogeneous traffic patterns.

Reference: Bai et al., "Adaptive Graph Convolutional Recurrent Network
for Traffic Forecasting", NeurIPS 2020.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AVWGCN(nn.Module):
    """Adaptive Virtual Weight Graph Convolution.

    Two key innovations:
    1. Learned adjacency: A = softmax(E @ E^T) — no predefined graph needed
    2. Node-adaptive weights: W_i = E_i @ W_pool — each node gets its own transform

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_nodes: Number of nodes in the graph
        embed_dim: Dimension of node embeddings
    """

    def __init__(self, in_features: int, out_features: int,
                 num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # Node embeddings for adaptive adjacency and node-adaptive weights
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim) * 0.1
        )

        # Weight pool: shared across nodes, combined with node embeddings
        # Each node's weight = E_i @ W_pool -> (in_features, out_features)
        self.weight_pool = nn.Parameter(
            torch.randn(embed_dim, in_features, out_features) * 0.1
        )
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, out_features))

    def forward(self, x: torch.Tensor,
                fixed_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, F_in) node features
            fixed_adj: Optional (N, N) fixed adjacency to blend with learned

        Returns:
            (B, N, F_out) transformed features
        """
        # Learned adjacency: softmax(E @ E^T)
        learned_adj = F.softmax(
            F.relu(torch.mm(self.node_embeddings, self.node_embeddings.T)),
            dim=1
        )

        # Optionally blend with fixed adjacency
        if fixed_adj is not None:
            adj = 0.5 * fixed_adj + 0.5 * learned_adj
        else:
            adj = learned_adj

        # Node-adaptive weights: W_i = E_i @ W_pool
        # node_embeddings: (N, embed_dim)
        # weight_pool: (embed_dim, F_in, F_out)
        # result: (N, F_in, F_out)
        node_weights = torch.einsum('ne,eio->nio', self.node_embeddings, self.weight_pool)
        node_bias = torch.matmul(self.node_embeddings, self.bias_pool)  # (N, F_out)

        # Apply node-adaptive transform: for each node, use its own weight
        # x: (B, N, F_in), node_weights: (N, F_in, F_out) -> (B, N, F_out)
        x_transformed = torch.einsum('bni,nio->bno', x, node_weights) + node_bias

        # Graph convolution: A @ X_transformed
        output = torch.matmul(adj, x_transformed)
        return output


class AGCRNCell(nn.Module):
    """AGCRN recurrent cell: AVWGCN inside GRU gates.

    Args:
        in_features: Input feature dimension per node
        hidden_dim: Hidden state dimension per node
        num_nodes: Number of nodes
        embed_dim: Node embedding dimension
    """

    def __init__(self, in_features: int, hidden_dim: int,
                 num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRU gates with AVWGCN
        self.gcn_z = AVWGCN(in_features + hidden_dim, hidden_dim, num_nodes, embed_dim)
        self.gcn_r = AVWGCN(in_features + hidden_dim, hidden_dim, num_nodes, embed_dim)
        self.gcn_h = AVWGCN(in_features + hidden_dim, hidden_dim, num_nodes, embed_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward one time step.

        Args:
            x: (B, N, F) input features at current time step
            h: (B, N, hidden_dim) previous hidden state
            adj: Optional (N, N) fixed adjacency matrix

        Returns:
            (B, N, hidden_dim) new hidden state
        """
        combined = torch.cat([x, h], dim=-1)

        z = torch.sigmoid(self.gcn_z(combined, adj))
        r = torch.sigmoid(self.gcn_r(combined, adj))

        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.gcn_h(combined_r, adj))

        h_new = z * h + (1 - z) * h_tilde
        return h_new


class AGCRN(nn.Module):
    """Adaptive Graph Convolutional Recurrent Network for traffic forecasting.

    Can work with or without a predefined adjacency matrix — learns its
    own graph structure from data via node embeddings.

    Args:
        in_features: Input features per node per time step
        hidden_dim: Hidden dimension of recurrent cells
        forecast_steps: Number of future steps to predict
        num_nodes: Number of nodes (parameterized for subgraph training)
        embed_dim: Node embedding dimension for adaptive adjacency
        num_layers: Number of stacked AGCRN layers
        dropout: Dropout rate
    """

    def __init__(self, in_features: int = 1, hidden_dim: int = 64,
                 forecast_steps: int = 12, num_nodes: int = 50,
                 embed_dim: int = 10, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Stacked AGCRN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = in_features if i == 0 else hidden_dim
            self.cells.append(
                AGCRNCell(cell_input_dim, hidden_dim, num_nodes, embed_dim)
            )

        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )

    def forward(self, x: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass — raw output (unbounded).

        Args:
            x: (B, N, T_in, F) input time-series
            adj: Optional (N, N) fixed adjacency matrix

        Returns:
            (B, N, T_out) raw forecast scores
        """
        B, N, T, F = x.shape

        # Initialize hidden states for each layer
        hidden_states = [
            torch.zeros(B, N, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        # Process each time step
        for t in range(T):
            x_t = x[:, :, t, :]  # (B, N, F)

            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    cell_input = x_t
                else:
                    cell_input = hidden_states[layer_idx - 1]

                hidden_states[layer_idx] = cell(
                    cell_input, hidden_states[layer_idx], adj
                )

        # Use top layer hidden state for prediction
        h_final = hidden_states[-1]  # (B, N, hidden_dim)
        output = self.output_fc(h_final)  # (B, N, forecast_steps)
        return output

    def forward_predict(self, x: torch.Tensor,
                        adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sigmoid bounding to [0, 1].

        Args:
            x: (B, N, T_in, F) input time-series
            adj: Optional (N, N) fixed adjacency matrix

        Returns:
            (B, N, T_out) forecast scores in [0, 1]
        """
        return torch.sigmoid(self.forward(x, adj))

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
