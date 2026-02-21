"""Model architectures for federated traffic congestion forecasting."""
from .camera_cnn import CameraCNN
from .fusion_model import FusionModel, LateFusionModel, GraphFusionModel
from .graph_forecaster import TGCN, GraphConvolution, TGCNCell, normalize_adjacency
from .agcrn import AGCRN, AVWGCN, AGCRNCell
from .adaptive_graph import AdaptiveAdjacency
from .baselines import HistoricalAverage, PersistenceModel, PerNodeLSTM, VARModel

__all__ = [
    # Camera
    'CameraCNN',
    # Fusion
    'FusionModel', 'LateFusionModel', 'GraphFusionModel',
    # Graph forecasting
    'TGCN', 'GraphConvolution', 'TGCNCell', 'normalize_adjacency',
    'AGCRN', 'AVWGCN', 'AGCRNCell',
    'AdaptiveAdjacency',
    # Baselines
    'HistoricalAverage', 'PersistenceModel', 'PerNodeLSTM', 'VARModel',
]
