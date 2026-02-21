"""Evaluation metrics for congestion prediction."""
import torch
import numpy as np
from typing import Dict, List, Union


def compute_metrics(predictions: Union[torch.Tensor, np.ndarray, List],
                    targets: Union[torch.Tensor, np.ndarray, List]) -> Dict[str, float]:
    """Compute all regression metrics for congestion prediction.

    Args:
        predictions: Model predictions [0, 1]
        targets: Ground truth values [0, 1]

    Returns:
        Dict with metrics: mse, rmse, mae, r2, accuracy_10pct
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))

    # R-squared (Coefficient of Determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Accuracy within 10% (useful for regression as classification proxy)
    within_10pct = np.mean(np.abs(predictions - targets) <= 0.1)

    # Accuracy within 20%
    within_20pct = np.mean(np.abs(predictions - targets) <= 0.2)

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'accuracy_10pct': float(within_10pct),
        'accuracy_20pct': float(within_20pct)
    }


class MSELoss:
    """Mean Squared Error loss tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with batch of predictions."""
        mse = torch.mean((predictions - targets) ** 2).item()
        batch_size = predictions.size(0)
        self.total += mse * batch_size
        self.count += batch_size

    def compute(self) -> float:
        """Compute average MSE."""
        return self.total / self.count if self.count > 0 else 0.0


class MAELoss:
    """Mean Absolute Error loss tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with batch of predictions."""
        mae = torch.mean(torch.abs(predictions - targets)).item()
        batch_size = predictions.size(0)
        self.total += mae * batch_size
        self.count += batch_size

    def compute(self) -> float:
        """Compute average MAE."""
        return self.total / self.count if self.count > 0 else 0.0


class R2Score:
    """R-squared (Coefficient of Determination) tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with batch of predictions."""
        self.predictions.extend(predictions.detach().cpu().numpy().flatten().tolist())
        self.targets.extend(targets.detach().cpu().numpy().flatten().tolist())

    def compute(self) -> float:
        """Compute R-squared."""
        if len(self.predictions) == 0:
            return 0.0

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class MetricTracker:
    """Track multiple metrics during training."""

    def __init__(self, metric_names: List[str] = None):
        if metric_names is None:
            metric_names = ['mse', 'mae', 'r2']

        self.metric_names = metric_names
        self.history = {name: [] for name in metric_names}
        self.reset()

    def reset(self):
        """Reset all metric trackers."""
        self.mse = MSELoss()
        self.mae = MAELoss()
        self.r2 = R2Score()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update all metrics with batch."""
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        self.mse.update(predictions, targets)
        self.mae.update(predictions, targets)
        self.r2.update(predictions, targets)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'mse': self.mse.compute(),
            'mae': self.mae.compute(),
            'r2': self.r2.compute()
        }

    def log_epoch(self):
        """Log current metrics to history and reset."""
        metrics = self.compute()
        for name, value in metrics.items():
            if name in self.history:
                self.history[name].append(value)
        self.reset()
        return metrics

    def get_best(self, metric: str = 'mse', mode: str = 'min') -> float:
        """Get best metric value from history."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return float('inf') if mode == 'min' else float('-inf')

        if mode == 'min':
            return min(self.history[metric])
        else:
            return max(self.history[metric])


# ──────────────────────────────────────────────────────────────────────
# Graph forecasting metrics
# ──────────────────────────────────────────────────────────────────────

def compute_mape(predictions: Union[torch.Tensor, np.ndarray],
                 targets: Union[torch.Tensor, np.ndarray],
                 epsilon: float = 1e-5) -> float:
    """Mean Absolute Percentage Error.

    Standard metric in traffic forecasting literature (METR-LA, PEMS-BAY).
    Skips near-zero targets to avoid division explosion.

    Args:
        predictions: Model predictions
        targets: Ground truth
        epsilon: Minimum target value to include in MAPE calculation

    Returns:
        MAPE as a fraction (not percentage)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # Only compute on non-trivial targets
    mask = np.abs(targets) > epsilon
    if mask.sum() == 0:
        return 0.0

    return float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])))


def compute_per_horizon_metrics(predictions: Union[torch.Tensor, np.ndarray],
                                 targets: Union[torch.Tensor, np.ndarray],
                                 horizons: Dict[str, int] = None,
                                 step_minutes: int = 5
                                 ) -> Dict[str, Dict[str, float]]:
    """Compute MAE, RMSE, MAPE at specific forecast horizons.

    Standard reporting format: metrics at 15min, 30min, 60min.

    Args:
        predictions: (B, N, T_out) or (N, T_out) predictions
        targets: Same shape as predictions
        horizons: Dict of {label: num_steps}. Default: 15/30/60 min.
        step_minutes: Minutes per time step

    Returns:
        Dict of {horizon_label: {mae, rmse, mape}}
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if horizons is None:
        horizons = {}
        for mins in [15, 30, 60]:
            steps = mins // step_minutes
            if steps <= predictions.shape[-1]:
                horizons[f'{mins}min'] = steps

    results = {}
    for label, num_steps in horizons.items():
        # Take predictions/targets up to this horizon
        p = predictions[..., :num_steps].flatten()
        t = targets[..., :num_steps].flatten()

        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mape = compute_mape(p, t)

        results[label] = {'mae': mae, 'rmse': rmse, 'mape': mape}

    return results


def compute_per_node_metrics(predictions: Union[torch.Tensor, np.ndarray],
                              targets: Union[torch.Tensor, np.ndarray]
                              ) -> Dict[str, np.ndarray]:
    """Compute per-camera error for spatial analysis.

    Args:
        predictions: (B, N, T_out) predictions
        targets: (B, N, T_out) targets

    Returns:
        Dict with per-node arrays: mae (N,), rmse (N,), mape (N,)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # (B, N, T) -> per-node metrics
    N = predictions.shape[-2] if predictions.ndim >= 2 else 1

    per_node_mae = np.zeros(N)
    per_node_rmse = np.zeros(N)

    for n in range(N):
        if predictions.ndim == 3:
            p = predictions[:, n, :].flatten()
            t = targets[:, n, :].flatten()
        else:
            p = predictions[n, :].flatten()
            t = targets[n, :].flatten()

        per_node_mae[n] = np.mean(np.abs(p - t))
        per_node_rmse[n] = np.sqrt(np.mean((p - t) ** 2))

    return {
        'mae': per_node_mae,
        'rmse': per_node_rmse,
    }


def compute_congestion_classification(predictions: Union[torch.Tensor, np.ndarray],
                                       binary_labels: Union[torch.Tensor, np.ndarray],
                                       threshold: float = 0.5
                                       ) -> Dict[str, float]:
    """Validate regression predictions against free-flow/blocked ground truth.

    Binarizes continuous predictions at threshold and compares to
    binary traffic state labels.

    Args:
        predictions: Continuous congestion scores [0, 1]
        binary_labels: Binary labels (0=free-flow, 1=blocked)
        threshold: Cutoff for binarizing predictions

    Returns:
        Dict with accuracy, precision, recall, f1
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(binary_labels, torch.Tensor):
        binary_labels = binary_labels.detach().cpu().numpy()

    predictions = np.array(predictions).flatten()
    binary_labels = np.array(binary_labels).flatten()

    pred_binary = (predictions >= threshold).astype(int)
    labels = binary_labels.astype(int)

    accuracy = float(np.mean(pred_binary == labels))

    # Precision, recall, F1 for blocked class
    tp = float(np.sum((pred_binary == 1) & (labels == 1)))
    fp = float(np.sum((pred_binary == 1) & (labels == 0)))
    fn = float(np.sum((pred_binary == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
