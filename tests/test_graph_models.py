"""Tests for graph forecasting models: T-GCN, AGCRN, AdaptiveAdjacency, GraphTimeSeriesDataset."""
import torch
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    """Small 10-node graph for fast tests."""
    N = 10
    adj = torch.zeros(N, N)
    # Simple chain: 0->1->2->...->9
    for i in range(N - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


@pytest.fixture
def batch_input():
    """Batch of graph time-series: (B=4, N=10, T=3, F=1)."""
    return torch.randn(4, 10, 3, 1)


@pytest.fixture
def large_batch():
    """Larger batch: (B=4, N=50, T=6, F=1) for 30-min input."""
    return torch.randn(4, 50, 6, 1)


@pytest.fixture
def large_adj():
    """50-node adjacency."""
    adj = torch.eye(50)
    for i in range(49):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


# ──────────────────────────────────────────────────────────────────────
# GraphConvolution
# ──────────────────────────────────────────────────────────────────────

class TestGraphConvolution:
    def test_output_shape(self, small_graph):
        from models.graph_forecaster import GraphConvolution, normalize_adjacency
        gcn = GraphConvolution(in_features=8, out_features=16)
        adj = normalize_adjacency(small_graph)
        x = torch.randn(4, 10, 8)  # (B, N, F)
        out = gcn(x, adj)
        assert out.shape == (4, 10, 16)

    def test_no_bias(self, small_graph):
        from models.graph_forecaster import GraphConvolution, normalize_adjacency
        gcn = GraphConvolution(8, 16, bias=False)
        assert gcn.bias is None
        adj = normalize_adjacency(small_graph)
        out = gcn(torch.randn(2, 10, 8), adj)
        assert out.shape == (2, 10, 16)


# ──────────────────────────────────────────────────────────────────────
# normalize_adjacency
# ──────────────────────────────────────────────────────────────────────

class TestNormalizeAdjacency:
    def test_with_self_loops(self, small_graph):
        from models.graph_forecaster import normalize_adjacency
        norm = normalize_adjacency(small_graph, add_self_loops=True)
        assert norm.shape == small_graph.shape
        # Diagonal should be non-zero (self-loops added)
        assert norm.diag().sum() > 0

    def test_symmetric(self, small_graph):
        from models.graph_forecaster import normalize_adjacency
        norm = normalize_adjacency(small_graph)
        # D^{-1/2} A D^{-1/2} should be symmetric if A is symmetric
        assert torch.allclose(norm, norm.T, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# TGCNCell
# ──────────────────────────────────────────────────────────────────────

class TestTGCNCell:
    def test_hidden_state_shape(self, small_graph):
        from models.graph_forecaster import TGCNCell, normalize_adjacency
        cell = TGCNCell(in_features=1, hidden_dim=32)
        adj = normalize_adjacency(small_graph)
        x = torch.randn(4, 10, 1)
        h = torch.zeros(4, 10, 32)
        h_new = cell(x, h, adj)
        assert h_new.shape == (4, 10, 32)

    def test_hidden_changes(self, small_graph):
        from models.graph_forecaster import TGCNCell, normalize_adjacency
        cell = TGCNCell(1, 16)
        adj = normalize_adjacency(small_graph)
        x = torch.randn(2, 10, 1)
        h = torch.zeros(2, 10, 16)
        h_new = cell(x, h, adj)
        # Hidden state should change from zero init
        assert not torch.allclose(h_new, h)


# ──────────────────────────────────────────────────────────────────────
# TGCN
# ──────────────────────────────────────────────────────────────────────

class TestTGCN:
    def test_output_shape_15min(self, batch_input, small_graph):
        """15-min forecast: 3 input steps -> 3 output steps."""
        from models.graph_forecaster import TGCN
        model = TGCN(in_features=1, hidden_dim=32, forecast_steps=3, num_nodes=10)
        out = model(batch_input, small_graph)
        assert out.shape == (4, 10, 3)

    def test_output_shape_30min(self, small_graph):
        from models.graph_forecaster import TGCN
        model = TGCN(in_features=1, hidden_dim=32, forecast_steps=6, num_nodes=10)
        x = torch.randn(4, 10, 6, 1)
        out = model(x, small_graph)
        assert out.shape == (4, 10, 6)

    def test_output_shape_60min(self, large_adj):
        from models.graph_forecaster import TGCN
        model = TGCN(in_features=1, hidden_dim=64, forecast_steps=12, num_nodes=50)
        x = torch.randn(4, 50, 12, 1)
        out = model(x, large_adj)
        assert out.shape == (4, 50, 12)

    def test_forward_predict_bounded(self, batch_input, small_graph):
        from models.graph_forecaster import TGCN
        model = TGCN(in_features=1, hidden_dim=32, forecast_steps=3, num_nodes=10)
        out = model.forward_predict(batch_input, small_graph)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_param_count(self):
        from models.graph_forecaster import TGCN
        model = TGCN(in_features=1, hidden_dim=32, forecast_steps=3, num_nodes=10)
        params = model.get_num_params()
        assert params > 0
        assert params < 100000  # Should be well under 100K

    def test_gradient_flow(self, batch_input, small_graph):
        from models.graph_forecaster import TGCN
        model = TGCN(in_features=1, hidden_dim=32, forecast_steps=3, num_nodes=10)
        out = model(batch_input, small_graph)
        loss = out.sum()
        loss.backward()
        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


# ──────────────────────────────────────────────────────────────────────
# AGCRN
# ──────────────────────────────────────────────────────────────────────

class TestAGCRN:
    def test_output_shape_no_adj(self, batch_input):
        """AGCRN should work without predefined adjacency."""
        from models.agcrn import AGCRN
        model = AGCRN(in_features=1, hidden_dim=32, forecast_steps=3,
                      num_nodes=10, embed_dim=8, num_layers=2)
        out = model(batch_input)
        assert out.shape == (4, 10, 3)

    def test_output_shape_with_adj(self, batch_input, small_graph):
        from models.agcrn import AGCRN
        model = AGCRN(in_features=1, hidden_dim=32, forecast_steps=3,
                      num_nodes=10, embed_dim=8)
        out = model(batch_input, small_graph)
        assert out.shape == (4, 10, 3)

    def test_forward_predict_bounded(self, batch_input):
        from models.agcrn import AGCRN
        model = AGCRN(in_features=1, hidden_dim=32, forecast_steps=3,
                      num_nodes=10, embed_dim=8)
        out = model.forward_predict(batch_input)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_multi_layer(self, batch_input):
        from models.agcrn import AGCRN
        for n_layers in [1, 2, 3]:
            model = AGCRN(in_features=1, hidden_dim=16, forecast_steps=3,
                          num_nodes=10, embed_dim=4, num_layers=n_layers)
            out = model(batch_input)
            assert out.shape == (4, 10, 3)

    def test_param_count(self):
        from models.agcrn import AGCRN
        model = AGCRN(in_features=1, hidden_dim=32, forecast_steps=3,
                      num_nodes=10, embed_dim=8, num_layers=2)
        params = model.get_num_params()
        assert params > 0

    def test_gradient_flow(self, batch_input):
        from models.agcrn import AGCRN
        model = AGCRN(in_features=1, hidden_dim=16, forecast_steps=3,
                      num_nodes=10, embed_dim=4)
        out = model(batch_input)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


# ──────────────────────────────────────────────────────────────────────
# AVWGCN
# ──────────────────────────────────────────────────────────────────────

class TestAVWGCN:
    def test_output_shape(self):
        from models.agcrn import AVWGCN
        gcn = AVWGCN(in_features=8, out_features=16, num_nodes=10, embed_dim=4)
        x = torch.randn(4, 10, 8)
        out = gcn(x)
        assert out.shape == (4, 10, 16)

    def test_with_fixed_adj(self, small_graph):
        from models.agcrn import AVWGCN
        gcn = AVWGCN(in_features=8, out_features=16, num_nodes=10, embed_dim=4)
        x = torch.randn(4, 10, 8)
        out = gcn(x, small_graph)
        assert out.shape == (4, 10, 16)


# ──────────────────────────────────────────────────────────────────────
# AdaptiveAdjacency
# ──────────────────────────────────────────────────────────────────────

class TestAdaptiveAdjacency:
    def test_output_shape(self, small_graph):
        from models.adaptive_graph import AdaptiveAdjacency
        ada = AdaptiveAdjacency(num_nodes=10, embed_dim=8)
        out = ada(small_graph)
        assert out.shape == (10, 10)

    def test_alpha_range(self):
        from models.adaptive_graph import AdaptiveAdjacency
        ada = AdaptiveAdjacency(num_nodes=10, alpha=0.3)
        assert 0.0 <= ada.alpha <= 1.0

    def test_learned_adjacency(self):
        from models.adaptive_graph import AdaptiveAdjacency
        ada = AdaptiveAdjacency(num_nodes=10, embed_dim=8)
        learned = ada.get_learned_adjacency()
        assert learned.shape == (10, 10)
        # Rows should sum to ~1 (softmax)
        row_sums = learned.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(10), atol=1e-5)

    def test_fixed_alpha(self, small_graph):
        from models.adaptive_graph import AdaptiveAdjacency
        ada = AdaptiveAdjacency(num_nodes=10, alpha=0.7, learn_alpha=False)
        assert ada.alpha == 0.7
        out = ada(small_graph)
        assert out.shape == (10, 10)

    def test_param_count(self):
        from models.adaptive_graph import AdaptiveAdjacency
        ada = AdaptiveAdjacency(num_nodes=10, embed_dim=8)
        params = ada.get_num_params()
        # 2 embeddings (10*8 each) + 1 alpha logit = 161
        assert params == 161


# ──────────────────────────────────────────────────────────────────────
# GraphTimeSeriesDataset
# ──────────────────────────────────────────────────────────────────────

class TestGraphTimeSeriesDataset:
    def test_basic_shape(self):
        from data.graph_dataset import GraphTimeSeriesDataset
        T, N, F = 100, 10, 1
        data = np.random.randn(T, N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3)

        x, y = ds[0]
        assert x.shape == (N, 3, F)
        assert y.shape == (N, 3)

    def test_length(self):
        from data.graph_dataset import GraphTimeSeriesDataset
        T, N, F = 100, 10, 1
        data = np.random.randn(T, N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3, stride=1)
        # T - (input + forecast) + 1 = 100 - 6 + 1 = 95
        assert len(ds) == 95

    def test_stride(self):
        from data.graph_dataset import GraphTimeSeriesDataset
        T, N, F = 100, 10, 1
        data = np.random.randn(T, N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3, stride=5)
        assert len(ds) == 19  # (100 - 6) // 5 + 1 = 19

    def test_subgraph(self):
        from data.graph_dataset import GraphTimeSeriesDataset
        T, N, F = 100, 20, 1
        data = np.random.randn(T, N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        node_subset = np.array([0, 3, 5, 7, 12])
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                     node_indices=node_subset)
        x, y = ds[0]
        assert x.shape == (5, 3, F)  # Only 5 nodes
        assert y.shape == (5, 3)
        assert ds.get_num_nodes() == 5
        assert ds.get_adjacency().shape == (5, 5)

    def test_target_is_future(self):
        """Verify target comes after input (no leakage)."""
        from data.graph_dataset import GraphTimeSeriesDataset
        T, N, F = 20, 3, 1
        # Linear ramp so we can verify ordering
        data = np.arange(T * N * F).reshape(T, N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3)

        x, y = ds[0]
        # Last input time step should be < first target time step
        # x: steps 0,1,2 for node 0 -> values 0,3,6 (stride N*F=3)
        # y: steps 3,4,5 for node 0 -> values 9,12,15
        assert x[0, -1, 0].item() < y[0, 0].item()

    def test_collate(self):
        from data.graph_dataset import GraphTimeSeriesDataset, graph_collate_fn
        from torch.utils.data import DataLoader
        T, N, F = 50, 5, 1
        data = np.random.randn(T, N, F).astype(np.float32)
        adj = np.eye(N, dtype=np.float32)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3)
        loader = DataLoader(ds, batch_size=8, collate_fn=graph_collate_fn)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (8, N, 3, F)
        assert y_batch.shape == (8, N, 3)
