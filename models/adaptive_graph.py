"""Learned adaptive adjacency module for T-GCN.

Blends a fixed (predefined) adjacency matrix with a learned one derived
from trainable node embeddings. Only used with T-GCN — AGCRN has its
own built-in adaptive adjacency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):
    """Learnable adjacency matrix that blends with a fixed graph.

    Learns two node embedding matrices E1, E2 and computes:
        A_learned = softmax(ReLU(E1 @ E2^T))
        A_final = alpha * A_fixed + (1 - alpha) * A_learned

    Args:
        num_nodes: Number of nodes in the graph
        embed_dim: Dimension of node embeddings
        alpha: Initial blend weight for fixed adjacency (0=fully learned, 1=fully fixed)
        learn_alpha: Whether alpha is a learnable parameter
    """

    def __init__(self, num_nodes: int, embed_dim: int = 16,
                 alpha: float = 0.5, learn_alpha: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # Two separate embeddings for asymmetric learned adjacency
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        if learn_alpha:
            # Learnable blend weight (stored as logit, sigmoid-bounded)
            init_logit = torch.log(torch.tensor(alpha / (1 - alpha + 1e-8)))
            self._alpha_logit = nn.Parameter(init_logit.detach().clone())
        else:
            self.register_buffer('_alpha_logit', None)
            self._alpha_fixed = alpha

    @property
    def alpha(self) -> float:
        """Current blend weight for fixed adjacency."""
        if self._alpha_logit is not None:
            return torch.sigmoid(self._alpha_logit).item()
        return self._alpha_fixed

    def forward(self, fixed_adj: torch.Tensor) -> torch.Tensor:
        """Compute blended adjacency matrix.

        Args:
            fixed_adj: (N, N) fixed adjacency matrix

        Returns:
            (N, N) blended adjacency matrix
        """
        # Learned adjacency: softmax(ReLU(E1 @ E2^T))
        learned_adj = F.softmax(
            F.relu(torch.mm(self.E1, self.E2.t())),
            dim=1
        )

        # Blend
        if self._alpha_logit is not None:
            alpha = torch.sigmoid(self._alpha_logit)
        else:
            alpha = self._alpha_fixed

        return alpha * fixed_adj + (1 - alpha) * learned_adj

    def get_learned_adjacency(self) -> torch.Tensor:
        """Return just the learned adjacency (for analysis)."""
        with torch.no_grad():
            return F.softmax(
                F.relu(torch.mm(self.E1, self.E2.t())),
                dim=1
            )

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
