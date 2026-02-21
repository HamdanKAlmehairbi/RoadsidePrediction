"""Federated learning infrastructure for traffic congestion forecasting."""
from .client import FLClient
from .camera_client import CameraClient
from .graph_client import GraphClient
from .server import FLServer
from .fedavg import fedavg_aggregate

__all__ = [
    'FLClient',
    'CameraClient',
    'GraphClient',
    'FLServer',
    'fedavg_aggregate',
]
