"""Ablation study on STREETS dataset.

Compares across multiple dimensions:
1. HA baseline (floor)
2. Persistence baseline (short-horizon floor)
3. Per-node LSTM (no graph, no FL)
4. VAR (classical multivariate)
5. T-GCN centralized (fixed graph)
6. AGCRN centralized (learned graph)
7. AGCRN federated, 2 clients
8. AGCRN federated, 10 clients
9. AGCRN federated, 20 clients

Each reported at 15-min, 30-min, 60-min horizons with MAE, RMSE, MAPE.
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
    partition_by_camera_subgroups,
)
from data.graph_dataset import GraphTimeSeriesDataset, graph_collate_fn
from models.graph_forecaster import TGCN
from models.agcrn import AGCRN
from models.baselines import HistoricalAverage, PersistenceModel, PerNodeLSTM, VARModel
from federated.graph_client import GraphClient
from federated.server import FLServer
from training.graph_trainer import GraphCentralizedTrainer
from training.graph_federated_trainer import GraphFederatedTrainer
from training.metrics import compute_metrics, compute_per_horizon_metrics


def evaluate_model(model, test_loader, adj_tensor, device, step_minutes=5):
    """Evaluate any model with common interface."""
    model.eval()
    model = model.to(device)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model.forward_predict(x_batch, adj_tensor)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    overall = compute_metrics(preds.flatten(), targets.flatten())
    per_horizon = compute_per_horizon_metrics(preds, targets, step_minutes=step_minutes)

    return {'overall': overall, 'per_horizon': per_horizon}


def run_federated_ablation(model_class, model_kwargs, graph, train_ts, test_ts,
                            camera_ids, num_clients, config, device):
    """Run federated training with a specific number of clients."""
    fl_cfg = config['federated']
    graph_cfg = config['graph']
    adj = graph['adjacency']
    N = len(camera_ids)

    # Create clients
    client_partitions = partition_by_camera_subgroups(camera_ids, num_clients)
    cam_to_idx = {c: i for i, c in enumerate(camera_ids)}

    clients = []
    for cid, client_cams in client_partitions.items():
        indices = [cam_to_idx[c] for c in client_cams]
        sub_adj = adj[np.ix_(indices, indices)]
        sub_ts = train_ts[:, indices, :]

        dataset = GraphTimeSeriesDataset(
            sub_ts, sub_adj,
            graph_cfg['input_steps'], graph_cfg['forecast_steps']
        )

        global_model = model_class(**model_kwargs)
        client = GraphClient(
            client_id=f"client_{cid}",
            model=global_model,
            dataset=dataset,
            adjacency=torch.FloatTensor(sub_adj),
            device=device,
            batch_size=fl_cfg['batch_size'],
            lr=fl_cfg['lr'],
        )
        clients.append(client)

    # Server
    global_model = model_class(**model_kwargs)
    server = FLServer(global_model)

    adj_tensor = torch.FloatTensor(adj).to(device)
    test_ds = GraphTimeSeriesDataset(
        test_ts, adj,
        graph_cfg['input_steps'], graph_cfg['forecast_steps']
    )
    test_loader = DataLoader(test_ds, batch_size=fl_cfg['batch_size'],
                             collate_fn=graph_collate_fn)

    trainer = GraphFederatedTrainer(
        server, clients, test_loader, adj_tensor, device=device
    )
    trainer.train(rounds=fl_cfg['rounds'], local_epochs=fl_cfg['local_epochs'])

    return server.evaluate_global_graph(test_loader, adj_tensor)


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
    batch_size = config['federated']['batch_size']

    train_ds = GraphTimeSeriesDataset(train_ts, adj, input_steps, forecast_steps)
    test_ds = GraphTimeSeriesDataset(test_ts, adj, input_steps, forecast_steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=graph_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             collate_fn=graph_collate_fn)

    adj_tensor = torch.FloatTensor(adj).to(device)

    results = {}

    # ── 1. Historical Average ──
    print("\n[1/9] Historical Average")
    ha = HistoricalAverage(forecast_steps=forecast_steps)
    results['1_ha'] = evaluate_model(ha, test_loader, adj_tensor, device)
    print(f"  {results['1_ha']['overall']}")

    # ── 2. Persistence ──
    print("\n[2/9] Persistence")
    persist = PersistenceModel(forecast_steps=forecast_steps)
    results['2_persistence'] = evaluate_model(persist, test_loader, adj_tensor, device)
    print(f"  {results['2_persistence']['overall']}")

    # ── 3. Per-Node LSTM ──
    print("\n[3/9] Per-Node LSTM")
    lstm = PerNodeLSTM(input_dim=graph_cfg['input_features'],
                       hidden_dim=32, forecast_steps=forecast_steps, num_nodes=N)
    print(f"  Params: {lstm.get_num_params()}")
    trainer = GraphCentralizedTrainer(lstm, adj_tensor, train_loader, test_loader,
                                      device=device, lr=config['federated']['lr'])
    trainer.train(epochs=30, early_stopping_patience=10)
    results['3_lstm'] = trainer.evaluate(test_loader)
    print(f"  {results['3_lstm']['overall']}")

    # ── 4. VAR ──
    print("\n[4/9] VAR")
    var = VARModel(num_nodes=N, lags=input_steps,
                   forecast_steps=forecast_steps, input_dim=graph_cfg['input_features'])
    print(f"  Params: {var.get_num_params()}")
    trainer = GraphCentralizedTrainer(var, adj_tensor, train_loader, test_loader,
                                      device=device, lr=config['federated']['lr'])
    trainer.train(epochs=30, early_stopping_patience=10)
    results['4_var'] = trainer.evaluate(test_loader)
    print(f"  {results['4_var']['overall']}")

    # ── 5. T-GCN Centralized ──
    print("\n[5/9] T-GCN Centralized")
    tgcn = TGCN(in_features=graph_cfg['input_features'],
                hidden_dim=graph_cfg['hidden_dim'],
                forecast_steps=forecast_steps, num_nodes=N,
                dropout=graph_cfg['dropout'])
    print(f"  Params: {tgcn.get_num_params()}")
    trainer = GraphCentralizedTrainer(tgcn, adj_tensor, train_loader, test_loader,
                                      device=device, lr=config['federated']['lr'])
    trainer.train(epochs=30, early_stopping_patience=10)
    results['5_tgcn_centralized'] = trainer.evaluate(test_loader)
    print(f"  {results['5_tgcn_centralized']['overall']}")

    # ── 6. AGCRN Centralized ──
    print("\n[6/9] AGCRN Centralized")
    agcrn = AGCRN(in_features=graph_cfg['input_features'],
                  hidden_dim=graph_cfg['hidden_dim'],
                  forecast_steps=forecast_steps, num_nodes=N,
                  embed_dim=graph_cfg['agcrn_embed_dim'],
                  num_layers=graph_cfg['agcrn_num_layers'],
                  dropout=graph_cfg['dropout'])
    print(f"  Params: {agcrn.get_num_params()}")
    trainer = GraphCentralizedTrainer(agcrn, adj_tensor, train_loader, test_loader,
                                      device=device, lr=config['federated']['lr'])
    trainer.train(epochs=30, early_stopping_patience=10)
    results['6_agcrn_centralized'] = trainer.evaluate(test_loader)
    print(f"  {results['6_agcrn_centralized']['overall']}")

    # ── 7-9. AGCRN Federated with varying clients ──
    agcrn_kwargs = dict(
        in_features=graph_cfg['input_features'],
        hidden_dim=graph_cfg['hidden_dim'],
        forecast_steps=forecast_steps,
        num_nodes=N,
        embed_dim=graph_cfg['agcrn_embed_dim'],
        num_layers=graph_cfg['agcrn_num_layers'],
        dropout=graph_cfg['dropout'],
    )

    for i, nc in enumerate([2, 10, 20], start=7):
        print(f"\n[{i}/9] AGCRN Federated ({nc} clients)")
        eval_results = run_federated_ablation(
            AGCRN, agcrn_kwargs, graph, train_ts, test_ts,
            camera_ids, nc, config, device
        )
        results[f'{i}_agcrn_fed_{nc}'] = {
            'overall': eval_results['overall'],
            'per_horizon': eval_results.get('per_horizon', {}),
            'num_clients': nc,
        }
        print(f"  {eval_results['overall']}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 70)
    for key, res in results.items():
        overall = res.get('overall', {})
        name = key.split('_', 1)[1] if '_' in key else key
        mae = overall.get('mae', float('nan'))
        rmse = overall.get('rmse', float('nan'))
        r2 = overall.get('r2', float('nan'))
        print(f"{name:<30} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")
    print("=" * 70)

    # Save results
    os.makedirs(config['experiments']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['experiments']['output_dir'], 'ablation_results.json')

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
