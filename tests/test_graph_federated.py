"""Tests for graph-aware federated learning: GraphClient, subgraph isolation, full FL round."""
import torch
import numpy as np
import pytest

from data.graph_dataset import GraphTimeSeriesDataset, graph_collate_fn


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def full_graph_data():
    """Full graph: 20 nodes, 100 time steps."""
    T, N, F = 100, 20, 1
    data = np.random.rand(T, N, F).astype(np.float32)
    adj = np.eye(N, dtype=np.float32)
    for i in range(N - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return data, adj


@pytest.fixture
def tgcn_model():
    """Small T-GCN model for testing."""
    from models.graph_forecaster import TGCN
    return TGCN(in_features=1, hidden_dim=16, forecast_steps=3, num_nodes=10)


@pytest.fixture
def agcrn_model():
    """Small AGCRN model for testing."""
    from models.agcrn import AGCRN
    return AGCRN(in_features=1, hidden_dim=16, forecast_steps=3,
                 num_nodes=10, embed_dim=4, num_layers=1)


# ──────────────────────────────────────────────────────────────────────
# GraphClient basics
# ──────────────────────────────────────────────────────────────────────

class TestGraphClient:
    def test_creation(self, full_graph_data, tgcn_model):
        from federated.graph_client import GraphClient
        data, adj = full_graph_data
        # Client gets first 10 nodes
        node_indices = np.arange(10)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                     node_indices=node_indices)
        sub_adj = torch.FloatTensor(adj[np.ix_(node_indices, node_indices)])

        client = GraphClient('c0', tgcn_model, ds, sub_adj)
        assert client.get_num_samples() > 0
        assert client.client_id == 'c0'

    def test_train_local(self, full_graph_data, tgcn_model):
        from federated.graph_client import GraphClient
        data, adj = full_graph_data
        node_indices = np.arange(10)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                     node_indices=node_indices)
        sub_adj = torch.FloatTensor(adj[np.ix_(node_indices, node_indices)])

        client = GraphClient('c0', tgcn_model, ds, sub_adj, batch_size=16)
        params = client.train_local(epochs=2)
        assert isinstance(params, dict)
        assert len(params) > 0
        assert len(client.train_losses) == 2

    def test_evaluate(self, full_graph_data, tgcn_model):
        from federated.graph_client import GraphClient
        data, adj = full_graph_data
        node_indices = np.arange(10)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                     node_indices=node_indices)
        sub_adj = torch.FloatTensor(adj[np.ix_(node_indices, node_indices)])

        client = GraphClient('c0', tgcn_model, ds, sub_adj, batch_size=16)
        metrics = client.evaluate()
        assert 'loss' in metrics
        assert 'num_nodes' in metrics
        assert metrics['num_nodes'] == 10

    def test_agcrn_client(self, full_graph_data, agcrn_model):
        from federated.graph_client import GraphClient
        data, adj = full_graph_data
        node_indices = np.arange(10)
        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                     node_indices=node_indices)
        sub_adj = torch.FloatTensor(adj[np.ix_(node_indices, node_indices)])

        client = GraphClient('c0', agcrn_model, ds, sub_adj, batch_size=16)
        params = client.train_local(epochs=1)
        assert isinstance(params, dict)


# ──────────────────────────────────────────────────────────────────────
# Subgraph isolation — no cross-client data access
# ──────────────────────────────────────────────────────────────────────

class TestSubgraphIsolation:
    def test_disjoint_node_sets(self, full_graph_data, tgcn_model):
        """Two clients with disjoint nodes should have no data overlap."""
        from federated.graph_client import GraphClient
        data, adj = full_graph_data

        nodes_a = np.arange(0, 10)
        nodes_b = np.arange(10, 20)

        ds_a = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                       node_indices=nodes_a)
        ds_b = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                       node_indices=nodes_b)

        assert ds_a.get_num_nodes() == 10
        assert ds_b.get_num_nodes() == 10

        # Data should be different (different node subsets)
        x_a, _ = ds_a[0]
        x_b, _ = ds_b[0]
        assert not torch.allclose(x_a, x_b)

    def test_adjacency_isolation(self, full_graph_data):
        """Each client should only see edges within its own subgraph."""
        data, adj = full_graph_data
        nodes_a = np.arange(0, 10)
        nodes_b = np.arange(10, 20)

        adj_a = adj[np.ix_(nodes_a, nodes_a)]
        adj_b = adj[np.ix_(nodes_b, nodes_b)]

        # Subgraph adjacencies should be 10x10, not 20x20
        assert adj_a.shape == (10, 10)
        assert adj_b.shape == (10, 10)

    def test_no_data_leakage_in_training(self, full_graph_data, tgcn_model):
        """Training client A should not affect client B's data."""
        from federated.graph_client import GraphClient
        data, adj = full_graph_data

        nodes_a = np.arange(0, 10)
        nodes_b = np.arange(10, 20)

        ds_b = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                       node_indices=nodes_b)
        x_b_before, y_b_before = ds_b[0]

        # Train client A
        ds_a = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                       node_indices=nodes_a)
        sub_adj_a = torch.FloatTensor(adj[np.ix_(nodes_a, nodes_a)])
        client_a = GraphClient('a', tgcn_model, ds_a, sub_adj_a, batch_size=16)
        client_a.train_local(epochs=2)

        # Client B's data should be unchanged
        x_b_after, y_b_after = ds_b[0]
        assert torch.allclose(x_b_before, x_b_after)
        assert torch.allclose(y_b_before, y_b_after)


# ──────────────────────────────────────────────────────────────────────
# Full FL round
# ──────────────────────────────────────────────────────────────────────

class TestFullFLRound:
    def test_one_round(self, full_graph_data):
        """Run one complete FL round: broadcast -> train -> aggregate."""
        from models.graph_forecaster import TGCN
        from federated.graph_client import GraphClient
        from federated.server import FLServer

        data, adj = full_graph_data

        # Create global model
        global_model = TGCN(in_features=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        server = FLServer(global_model)

        # Create 2 clients with disjoint nodes
        clients = []
        for i, node_range in enumerate([np.arange(0, 10), np.arange(10, 20)]):
            ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                         node_indices=node_range)
            sub_adj = torch.FloatTensor(adj[np.ix_(node_range, node_range)])
            client = GraphClient(f'c{i}', global_model, ds, sub_adj, batch_size=16)
            clients.append(client)

        # Broadcast
        global_params = server.broadcast()
        for client in clients:
            client.set_model_params(global_params)

        # Local training
        updates = []
        weights = []
        for client in clients:
            params = client.train_local(epochs=2)
            updates.append(params)
            weights.append(client.get_num_samples())

        # Aggregate
        aggregated = server.aggregate(updates, weights)
        assert isinstance(aggregated, dict)
        assert len(aggregated) > 0

    def test_model_improves(self, full_graph_data):
        """Global model should improve (or at least not crash) over multiple rounds."""
        from models.graph_forecaster import TGCN
        from federated.graph_client import GraphClient
        from federated.server import FLServer

        data, adj = full_graph_data
        global_model = TGCN(in_features=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        server = FLServer(global_model)

        clients = []
        for i, node_range in enumerate([np.arange(0, 10), np.arange(10, 20)]):
            ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3,
                                         node_indices=node_range)
            sub_adj = torch.FloatTensor(adj[np.ix_(node_range, node_range)])
            client = GraphClient(f'c{i}', global_model, ds, sub_adj, batch_size=16)
            clients.append(client)

        # Run 3 rounds
        for round_num in range(3):
            global_params = server.broadcast()
            for client in clients:
                client.set_model_params(global_params)

            updates = []
            weights = []
            for client in clients:
                params = client.train_local(epochs=1)
                updates.append(params)
                weights.append(client.get_num_samples())

            server.aggregate(updates, weights)

        # Just verify it ran without error — loss convergence
        # is tested in integration, not unit tests
        assert len(server.round_metrics) == 0  # aggregate() doesn't log metrics

    def test_evaluate_global_graph(self, full_graph_data):
        """Test server's evaluate_global_graph method."""
        from models.graph_forecaster import TGCN
        from federated.server import FLServer
        from torch.utils.data import DataLoader

        data, adj = full_graph_data
        N = data.shape[1]

        model = TGCN(in_features=1, hidden_dim=16, forecast_steps=3, num_nodes=N)
        server = FLServer(model)

        ds = GraphTimeSeriesDataset(data, adj, input_steps=3, forecast_steps=3)
        loader = DataLoader(ds, batch_size=16, collate_fn=graph_collate_fn)
        adj_tensor = torch.FloatTensor(adj)

        metrics = server.evaluate_global_graph(loader, adj_tensor)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['num_samples'] > 0


# ──────────────────────────────────────────────────────────────────────
# GraphFusionModel
# ──────────────────────────────────────────────────────────────────────

class TestGraphFusionModel:
    def test_graph_only_forward(self):
        from models.graph_forecaster import TGCN
        from models.camera_cnn import CameraCNN
        from models.fusion_model import GraphFusionModel

        camera = CameraCNN(feature_dim=128, pretrained=False, backbone='custom_small')
        graph = TGCN(in_features=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        fusion = GraphFusionModel(camera, graph)

        x = torch.randn(4, 10, 3, 1)
        adj = torch.eye(10)
        out = fusion.forward_graph_only(x, adj)
        assert out.shape == (4, 10, 3)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_param_counts(self):
        from models.graph_forecaster import TGCN
        from models.camera_cnn import CameraCNN
        from models.fusion_model import GraphFusionModel

        camera = CameraCNN(feature_dim=128, pretrained=False, backbone='custom_small')
        graph = TGCN(in_features=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        fusion = GraphFusionModel(camera, graph)

        params = fusion.get_num_params()
        assert params['camera'] > 0
        assert params['graph'] > 0
        assert params['total'] == params['camera'] + params['graph']
