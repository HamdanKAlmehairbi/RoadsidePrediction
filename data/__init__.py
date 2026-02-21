"""Data pipeline for federated traffic congestion forecasting."""
from .camera_dataset import CameraDataset
from .graph_dataset import GraphTimeSeriesDataset, graph_collate_fn
from .streets_loader import (
    load_streets_graph,
    load_streets_traffic_counts,
    load_streets_traffic_state,
    compute_congestion_from_counts,
    build_node_timeseries,
    load_streets_image_index,
    extract_streets_camera_samples,
    partition_by_community,
    partition_by_camera_subgroups,
    split_by_year,
    get_streets_dataset_stats,
)

__all__ = [
    # Datasets
    'CameraDataset',
    'GraphTimeSeriesDataset',
    'graph_collate_fn',
    # STREETS loaders
    'load_streets_graph',
    'load_streets_traffic_counts',
    'load_streets_traffic_state',
    'compute_congestion_from_counts',
    'build_node_timeseries',
    'load_streets_image_index',
    'extract_streets_camera_samples',
    'partition_by_community',
    'partition_by_camera_subgroups',
    'split_by_year',
    'get_streets_dataset_stats',
]
