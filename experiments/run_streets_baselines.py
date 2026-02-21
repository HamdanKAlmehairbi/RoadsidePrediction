"""Baseline models on STREETS dataset.

Runs HA, Persistence, Per-Node LSTM, and VAR as comparison baselines.
Reports metrics at 15min horizon.
"""
import torch
import numpy as np
import yaml
import os
import json
from torch.utils.data import DataLoader

from data.streets_loader import (
    load_streets_graph, load_streets_traffic_counts,
    load_streets_traffic_state, compute_congestion_from_counts,
    build_node_timeseries, split_by_year,
)
from data.graph_dataset import GraphTimeSeriesDataset, graph_collate_fn
from models.baselines import HistoricalAverage, PersistenceModel, PerNodeLSTM, VARModel
from training.graph_trainer import GraphCentralizedTrainer
from training.metrics import compute_metrics, compute_per_horizon_metrics


def evaluate_baseline(model, test_loader, device, step_minutes=5):
    """Evaluate a non-trainable or trained baseline model."""
    model.eval()
    model = model.to(device)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model.forward_predict(x_batch)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    overall = compute_metrics(preds.flatten(), targets.flatten())
    per_horizon = compute_per_horizon_metrics(preds, targets, step_minutes=step_minutes)

    return {'overall': overall, 'per_horizon': per_horizon}


def main():
    with open('configs/default_config.yaml') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['experiments']['seed'])
    np.random.seed(config['experiments']['seed'])

    device = config['experiments']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    streets_cfg = config['data']['streets']
    graph_cfg = config['graph']

    # Load data
    print("Loading STREETS data...")
    graph = load_streets_graph(streets_cfg['dataroot'])
    counts = load_streets_traffic_counts(streets_cfg['dataroot'])

    try:
        states = load_streets_traffic_state(streets_cfg['dataroot'])
    except FileNotFoundError:
        states = None

    congestion = compute_congestion_from_counts(
        counts, states, max_count=streets_cfg['max_count_normalization']
    )

    train_data = split_by_year(congestion, streets_cfg['train_year'])
    test_data = split_by_year(congestion, streets_cfg['test_year'])

    camera_ids = graph['camera_ids']
    adj = graph['adjacency']
    N = len(camera_ids)

    print(f"Cameras: {N}, Train: {len(train_data)}, Test: {len(test_data)}")

    train_ts = build_node_timeseries(train_data, camera_ids, freq=streets_cfg['train_freq'])
    test_ts = build_node_timeseries(test_data, camera_ids, freq=streets_cfg['test_freq'])

    input_steps = graph_cfg['input_steps']
    forecast_steps = graph_cfg['forecast_steps']

    # Create datasets
    train_ds = GraphTimeSeriesDataset(train_ts, adj, input_steps, forecast_steps)
    test_ds = GraphTimeSeriesDataset(test_ts, adj, input_steps, forecast_steps)

    batch_size = config['federated']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=graph_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             collate_fn=graph_collate_fn)

    results = {}

    # 1. Historical Average
    print("\n=== Historical Average ===")
    ha = HistoricalAverage(forecast_steps=forecast_steps)
    results['historical_average'] = evaluate_baseline(ha, test_loader, device)
    print(f"HA: {results['historical_average']['overall']}")

    # 2. Persistence
    print("\n=== Persistence ===")
    persist = PersistenceModel(forecast_steps=forecast_steps)
    results['persistence'] = evaluate_baseline(persist, test_loader, device)
    print(f"Persistence: {results['persistence']['overall']}")

    # 3. Per-Node LSTM
    print("\n=== Per-Node LSTM ===")
    lstm = PerNodeLSTM(
        input_dim=graph_cfg['input_features'],
        hidden_dim=32,
        forecast_steps=forecast_steps,
        num_nodes=N,
    )
    print(f"Parameters: {lstm.get_num_params()}")

    # Train LSTM using graph trainer (adjacency unused by LSTM)
    adj_tensor = torch.FloatTensor(adj)
    trainer = GraphCentralizedTrainer(
        lstm, adj_tensor, train_loader, test_loader,
        device=device, lr=config['federated']['lr']
    )
    trainer.train(epochs=30, early_stopping_patience=10)
    results['per_node_lstm'] = trainer.evaluate(test_loader)
    print(f"LSTM: {results['per_node_lstm']['overall']}")

    # 4. VAR Model
    print("\n=== VAR Model ===")
    var = VARModel(
        num_nodes=N,
        lags=input_steps,
        forecast_steps=forecast_steps,
        input_dim=graph_cfg['input_features'],
    )
    print(f"Parameters: {var.get_num_params()}")

    trainer = GraphCentralizedTrainer(
        var, adj_tensor, train_loader, test_loader,
        device=device, lr=config['federated']['lr']
    )
    trainer.train(epochs=30, early_stopping_patience=10)
    results['var'] = trainer.evaluate(test_loader)
    print(f"VAR: {results['var']['overall']}")

    # Save results
    os.makedirs(config['experiments']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['experiments']['output_dir'], 'baseline_results.json')

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=to_serializable)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
