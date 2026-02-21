"""Centralized T-GCN + AGCRN baselines on STREETS dataset.

Trains on the full graph (all nodes) and single community (reduced graph).
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
from models.graph_forecaster import TGCN
from models.agcrn import AGCRN
from training.graph_trainer import GraphCentralizedTrainer


def train_and_evaluate(model, adj_tensor, train_loader, test_loader,
                       config, device, epochs=100, label=""):
    """Train a model and return results."""
    print(f"Parameters: {model.get_num_params()}")

    trainer = GraphCentralizedTrainer(
        model, adj_tensor, train_loader, test_loader,
        device=device, lr=config['federated']['lr'],
        weight_decay=config['training']['weight_decay'],
        scheduler='cosine',
        scheduler_params={'T_max': epochs},
    )
    history = trainer.train(epochs=epochs, early_stopping_patience=20)
    eval_results = trainer.evaluate(test_loader)
    print(f"{label}: {eval_results['overall']}")
    print(f"Per-horizon: {eval_results['per_horizon']}")
    return eval_results


def main():
    # Load config
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

    # Chronological split
    train_data = split_by_year(congestion, streets_cfg['train_year'])
    test_data = split_by_year(congestion, streets_cfg['test_year'])

    camera_ids = graph['camera_ids']
    adj = graph['adjacency']
    N = len(camera_ids)

    print(f"Cameras: {N}, Train: {len(train_data)}, Test: {len(test_data)}")

    # Build time-series
    train_ts = build_node_timeseries(train_data, camera_ids, freq=streets_cfg['train_freq'])
    test_ts = build_node_timeseries(test_data, camera_ids, freq=streets_cfg['test_freq'])

    input_steps = graph_cfg['input_steps']
    forecast_steps = graph_cfg['forecast_steps']
    batch_size = config['federated']['batch_size']
    epochs = 50

    results = {}

    # ════════════════════════════════════════════════
    # PART 1: Single community (Buffalo Grove only)
    # ════════════════════════════════════════════════
    comm_name = sorted(graph['communities'].keys())[0]  # buffalogrove
    comm_cams = graph['communities'][comm_name]
    comm_adj = graph['community_adj'][comm_name]
    cam_to_idx = {c: i for i, c in enumerate(camera_ids)}
    comm_indices = [cam_to_idx[c] for c in comm_cams]
    N_comm = len(comm_cams)

    comm_train_ts = train_ts[:, comm_indices, :]
    comm_test_ts = test_ts[:, comm_indices, :]

    comm_train_ds = GraphTimeSeriesDataset(comm_train_ts, comm_adj, input_steps, forecast_steps)
    comm_test_ds = GraphTimeSeriesDataset(comm_test_ts, comm_adj, input_steps, forecast_steps)

    comm_train_loader = DataLoader(comm_train_ds, batch_size=batch_size,
                                   shuffle=True, collate_fn=graph_collate_fn)
    comm_test_loader = DataLoader(comm_test_ds, batch_size=batch_size,
                                  collate_fn=graph_collate_fn)
    comm_adj_tensor = torch.FloatTensor(comm_adj)

    print(f"\n{'='*60}")
    print(f"SINGLE COMMUNITY: {comm_name} ({N_comm} cameras)")
    print(f"Input: {input_steps} steps, Forecast: {forecast_steps} steps")
    print(f"Epochs: {epochs}, Scheduler: cosine, Patience: 20")
    print(f"{'='*60}")

    # T-GCN on single community
    print(f"\n=== T-GCN ({comm_name}) ===")
    tgcn = TGCN(in_features=graph_cfg['input_features'],
                hidden_dim=graph_cfg['hidden_dim'],
                forecast_steps=forecast_steps,
                num_nodes=N_comm,
                dropout=graph_cfg['dropout'])
    results[f'tgcn_{comm_name}'] = train_and_evaluate(
        tgcn, comm_adj_tensor, comm_train_loader, comm_test_loader,
        config, device, epochs=epochs, label=f"T-GCN ({comm_name})"
    )

    # AGCRN on single community
    print(f"\n=== AGCRN ({comm_name}) ===")
    agcrn = AGCRN(in_features=graph_cfg['input_features'],
                  hidden_dim=graph_cfg['hidden_dim'],
                  forecast_steps=forecast_steps,
                  num_nodes=N_comm,
                  embed_dim=graph_cfg['agcrn_embed_dim'],
                  num_layers=graph_cfg['agcrn_num_layers'],
                  dropout=graph_cfg['dropout'])
    results[f'agcrn_{comm_name}'] = train_and_evaluate(
        agcrn, comm_adj_tensor, comm_train_loader, comm_test_loader,
        config, device, epochs=epochs, label=f"AGCRN ({comm_name})"
    )

    # ════════════════════════════════════════════════
    # PART 2: Full graph (all 320 cameras)
    # ════════════════════════════════════════════════
    train_ds = GraphTimeSeriesDataset(train_ts, adj, input_steps, forecast_steps)
    test_ds = GraphTimeSeriesDataset(test_ts, adj, input_steps, forecast_steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=graph_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             collate_fn=graph_collate_fn)
    adj_tensor = torch.FloatTensor(adj)

    print(f"\n{'='*60}")
    print(f"FULL GRAPH: all {N} cameras")
    print(f"Input: {input_steps} steps, Forecast: {forecast_steps} steps")
    print(f"Epochs: {epochs}, Scheduler: cosine, Patience: 20")
    print(f"{'='*60}")

    # T-GCN full graph
    print("\n=== T-GCN (full graph) ===")
    tgcn = TGCN(in_features=graph_cfg['input_features'],
                hidden_dim=graph_cfg['hidden_dim'],
                forecast_steps=forecast_steps,
                num_nodes=N,
                dropout=graph_cfg['dropout'])
    results['tgcn_full'] = train_and_evaluate(
        tgcn, adj_tensor, train_loader, test_loader,
        config, device, epochs=epochs, label="T-GCN (full)"
    )

    # AGCRN full graph
    print("\n=== AGCRN (full graph) ===")
    agcrn = AGCRN(in_features=graph_cfg['input_features'],
                  hidden_dim=graph_cfg['hidden_dim'],
                  forecast_steps=forecast_steps,
                  num_nodes=N,
                  embed_dim=graph_cfg['agcrn_embed_dim'],
                  num_layers=graph_cfg['agcrn_num_layers'],
                  dropout=graph_cfg['dropout'])
    results['agcrn_full'] = train_and_evaluate(
        agcrn, adj_tensor, train_loader, test_loader,
        config, device, epochs=epochs, label="AGCRN (full)"
    )

    # ════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"{'-'*70}")
    for key, res in results.items():
        overall = res.get('overall', {})
        mae = overall.get('mae', float('nan'))
        rmse = overall.get('rmse', float('nan'))
        r2 = overall.get('r2', float('nan'))
        print(f"{key:<30} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(config['experiments']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['experiments']['output_dir'], 'centralized_results_v2.json')

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
