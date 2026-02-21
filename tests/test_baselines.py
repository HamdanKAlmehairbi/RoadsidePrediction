"""Tests for baseline forecasting models and graph forecasting metrics."""
import torch
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_input():
    """Standard batch: (B=4, N=10, T_in=3, F=1) for 15-min input."""
    return torch.randn(4, 10, 3, 1)


@pytest.fixture
def constant_input():
    """Constant-value batch for testing persistence/HA correctness."""
    x = torch.ones(2, 5, 3, 1) * 0.4
    return x


# ──────────────────────────────────────────────────────────────────────
# HistoricalAverage
# ──────────────────────────────────────────────────────────────────────

class TestHistoricalAverage:
    def test_output_shape(self, batch_input):
        from models.baselines import HistoricalAverage
        model = HistoricalAverage(forecast_steps=3)
        out = model(batch_input)
        assert out.shape == (4, 10, 3)

    def test_predict_bounded(self, batch_input):
        from models.baselines import HistoricalAverage
        model = HistoricalAverage(forecast_steps=3)
        out = model.forward_predict(torch.abs(batch_input))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_constant_input(self, constant_input):
        from models.baselines import HistoricalAverage
        model = HistoricalAverage(forecast_steps=3)
        out = model(constant_input)
        # Mean of all 0.4s should be 0.4
        assert torch.allclose(out, torch.ones_like(out) * 0.4, atol=1e-6)

    def test_zero_params(self):
        from models.baselines import HistoricalAverage
        model = HistoricalAverage(forecast_steps=3)
        assert model.get_num_params() == 0


# ──────────────────────────────────────────────────────────────────────
# PersistenceModel
# ──────────────────────────────────────────────────────────────────────

class TestPersistenceModel:
    def test_output_shape(self, batch_input):
        from models.baselines import PersistenceModel
        model = PersistenceModel(forecast_steps=3)
        out = model(batch_input)
        assert out.shape == (4, 10, 3)

    def test_repeats_last_value(self):
        from models.baselines import PersistenceModel
        model = PersistenceModel(forecast_steps=3)
        # Known values: last time step has specific values
        x = torch.zeros(2, 5, 3, 1)
        x[:, :, -1, 0] = 0.7  # Last time step = 0.7
        out = model(x)
        expected = torch.ones(2, 5, 3) * 0.7
        assert torch.allclose(out, expected, atol=1e-6)

    def test_zero_params(self):
        from models.baselines import PersistenceModel
        assert PersistenceModel(forecast_steps=3).get_num_params() == 0


# ──────────────────────────────────────────────────────────────────────
# PerNodeLSTM
# ──────────────────────────────────────────────────────────────────────

class TestPerNodeLSTM:
    def test_output_shape(self, batch_input):
        from models.baselines import PerNodeLSTM
        model = PerNodeLSTM(input_dim=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        out = model(batch_input)
        assert out.shape == (4, 10, 3)

    def test_forward_predict_bounded(self, batch_input):
        from models.baselines import PerNodeLSTM
        model = PerNodeLSTM(input_dim=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        out = model.forward_predict(batch_input)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_ignores_adj(self, batch_input):
        from models.baselines import PerNodeLSTM
        model = PerNodeLSTM(input_dim=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        adj = torch.eye(10)
        out1 = model(batch_input, adj=adj)
        out2 = model(batch_input, adj=None)
        assert torch.allclose(out1, out2)

    def test_has_params(self):
        from models.baselines import PerNodeLSTM
        model = PerNodeLSTM(input_dim=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        assert model.get_num_params() > 0

    def test_gradient_flow(self, batch_input):
        from models.baselines import PerNodeLSTM
        model = PerNodeLSTM(input_dim=1, hidden_dim=16, forecast_steps=3, num_nodes=10)
        out = model(batch_input)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


# ──────────────────────────────────────────────────────────────────────
# VARModel
# ──────────────────────────────────────────────────────────────────────

class TestVARModel:
    def test_output_shape(self, batch_input):
        from models.baselines import VARModel
        model = VARModel(num_nodes=10, lags=3, forecast_steps=3, input_dim=1)
        out = model(batch_input)
        assert out.shape == (4, 10, 3)

    def test_forward_predict_bounded(self, batch_input):
        from models.baselines import VARModel
        model = VARModel(num_nodes=10, lags=3, forecast_steps=3, input_dim=1)
        out = model.forward_predict(batch_input)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_handles_short_input(self):
        """Input shorter than lags should still work (padding)."""
        from models.baselines import VARModel
        model = VARModel(num_nodes=5, lags=6, forecast_steps=3, input_dim=1)
        x = torch.randn(2, 5, 3, 1)  # Only 3 steps, lags=6
        out = model(x)
        assert out.shape == (2, 5, 3)


# ──────────────────────────────────────────────────────────────────────
# Graph forecasting metrics
# ──────────────────────────────────────────────────────────────────────

class TestMAPE:
    def test_perfect_prediction(self):
        from training.metrics import compute_mape
        pred = np.array([0.5, 0.3, 0.8])
        target = np.array([0.5, 0.3, 0.8])
        assert compute_mape(pred, target) == 0.0

    def test_known_value(self):
        from training.metrics import compute_mape
        pred = np.array([1.0])
        target = np.array([2.0])
        # |2.0 - 1.0| / 2.0 = 0.5
        assert abs(compute_mape(pred, target) - 0.5) < 1e-6

    def test_skips_near_zero(self):
        from training.metrics import compute_mape
        pred = np.array([0.1, 0.5])
        target = np.array([0.0, 0.5])  # First target is zero
        # Should only compute on second element (target=0.5)
        mape = compute_mape(pred, target, epsilon=1e-5)
        assert mape == 0.0  # pred=0.5, target=0.5 -> 0%


class TestPerHorizonMetrics:
    def test_output_structure(self):
        from training.metrics import compute_per_horizon_metrics
        pred = np.random.rand(4, 10, 6)
        target = np.random.rand(4, 10, 6)
        results = compute_per_horizon_metrics(pred, target, step_minutes=5)
        assert '15min' in results
        assert '30min' in results
        for horizon, metrics in results.items():
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'mape' in metrics

    def test_15min_only(self):
        from training.metrics import compute_per_horizon_metrics
        pred = np.random.rand(4, 10, 3)  # Only 3 steps
        target = np.random.rand(4, 10, 3)
        results = compute_per_horizon_metrics(pred, target, step_minutes=5)
        assert '15min' in results
        assert '30min' not in results  # Not enough steps


class TestPerNodeMetrics:
    def test_output_shape(self):
        from training.metrics import compute_per_node_metrics
        pred = np.random.rand(4, 10, 3)
        target = np.random.rand(4, 10, 3)
        results = compute_per_node_metrics(pred, target)
        assert results['mae'].shape == (10,)
        assert results['rmse'].shape == (10,)

    def test_perfect_node(self):
        from training.metrics import compute_per_node_metrics
        pred = np.random.rand(4, 10, 3)
        target = pred.copy()
        results = compute_per_node_metrics(pred, target)
        assert np.allclose(results['mae'], 0.0, atol=1e-6)


class TestCongestionClassification:
    def test_perfect_classification(self):
        from training.metrics import compute_congestion_classification
        pred = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = compute_congestion_classification(pred, labels, threshold=0.5)
        assert result['accuracy'] == 1.0
        assert result['f1'] == 1.0

    def test_all_wrong(self):
        from training.metrics import compute_congestion_classification
        pred = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([0, 0, 1, 1])
        result = compute_congestion_classification(pred, labels, threshold=0.5)
        assert result['accuracy'] == 0.0
