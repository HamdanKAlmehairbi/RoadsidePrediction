"""Federated T-GCN + AGCRN on STREETS dataset.

Trains with subgraph-based FL (10-20 clients per community).
Compares federated vs centralized performance (FL gap).
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
from federated.graph_client import GraphClient
from federated.server import FLServer
from training.graph_federated_trainer import GraphFederatedTrainer


def create_clients(model_class, model_kwargs, graph, train_ts,
                   camera_ids, num_clients, config, device):
    """Create FL clients with subgraph partitioning."""
    graph_cfg = config['graph']
    fl_cfg = config['federated']
    input_steps = graph_cfg['input_steps']
    forecast_steps = graph_cfg['forecast_steps']

    # Partition cameras across clients
    client_partitions = partition_by_camera_subgroups(camera_ids, num_clients)
    adj = graph['adjacency']
    cam_to_idx = {c: i for i, c in enumerate(camera_ids)}

    clients = []
    for cid, client_cams in client_partitions.items():
        # Extract subgraph for this client
        indices = [cam_to_idx[c] for c in client_cams]
        sub_adj = adj[np.ix_(indices, indices)]
        sub_ts = train_ts[:, indices, :]

        sub_adj_tensor = torch.FloatTensor(sub_adj)
        dataset = GraphTimeSeriesDataset(
            sub_ts, sub_adj, input_steps, forecast_steps
        )

        # Create global model template for weight sharing
        global_model = model_class(**model_kwargs)

        client = GraphClient(
            client_id=f"client_{cid}",
            model=global_model,
            dataset=dataset,
            adjacency=sub_adj_tensor,
            device=device,
            batch_size=fl_cfg['batch_size'],
            lr=fl_cfg['lr'],
        )
        clients.append(client)

    return clients


def run_federated(model_name, model_class, model_kwargs,
                  graph, train_ts, test_ts, camera_ids, config, device):
    """Run federated training for a given model."""
    fl_cfg = config['federated']
    graph_cfg = config['graph']

    num_clients = fl_cfg['num_graph_clients']
    adj = graph['adjacency']
    N = len(camera_ids)

    print(f"\n=== {model_name} Federated ({num_clients} clients) ===")

    # Create clients
    clients = create_clients(
        model_class, model_kwargs, graph, train_ts,
        camera_ids, num_clients, config, device
    )

    for i, c in enumerate(clients):
        print(f"  Client {i}: {c.dataset.get_num_nodes()} nodes, "
              f"{len(c.dataset)} samples")

    # Create server with global model
    global_model = model_class(**model_kwargs)
    server = FLServer(global_model)

    # Test loader (full graph)
    adj_tensor = torch.FloatTensor(adj)
    test_ds = GraphTimeSeriesDataset(
        test_ts, adj,
        graph_cfg['input_steps'], graph_cfg['forecast_steps']
    )
    test_loader = DataLoader(
        test_ds, batch_size=fl_cfg['batch_size'],
        collate_fn=graph_collate_fn
    )

    # Train
    trainer = GraphFederatedTrainer(
        server, clients, test_loader, adj_tensor, device=device
    )
    history = trainer.train(
        rounds=fl_cfg['rounds'],
        local_epochs=fl_cfg['local_epochs'],
    )

    # Final evaluation
    eval_results = server.evaluate_global_graph(test_loader, adj_tensor)

    return {
        'history': {k: [float(v) if v is not None else None for v in vals]
                    for k, vals in history.items()},
        'eval': eval_results,
        'num_clients': num_clients,
    }


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

    results = {}

    # Federated T-GCN
    tgcn_kwargs = dict(
        in_features=graph_cfg['input_features'],
        hidden_dim=graph_cfg['hidden_dim'],
        forecast_steps=forecast_steps,
        num_nodes=N,
        dropout=graph_cfg['dropout'],
    )
    results['tgcn_federated'] = run_federated(
        'T-GCN', TGCN, tgcn_kwargs,
        graph, train_ts, test_ts, camera_ids, config, device
    )
    print(f"T-GCN Federated: {results['tgcn_federated']['eval']['overall']}")

    # Federated AGCRN
    agcrn_kwargs = dict(
        in_features=graph_cfg['input_features'],
        hidden_dim=graph_cfg['hidden_dim'],
        forecast_steps=forecast_steps,
        num_nodes=N,
        embed_dim=graph_cfg['agcrn_embed_dim'],
        num_layers=graph_cfg['agcrn_num_layers'],
        dropout=graph_cfg['dropout'],
    )
    results['agcrn_federated'] = run_federated(
        'AGCRN', AGCRN, agcrn_kwargs,
        graph, train_ts, test_ts, camera_ids, config, device
    )
    print(f"AGCRN Federated: {results['agcrn_federated']['eval']['overall']}")

    # Save results
    os.makedirs(config['experiments']['output_dir'], exist_ok=True)
    output_path = os.path.join(config['experiments']['output_dir'], 'federated_results.json')

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
