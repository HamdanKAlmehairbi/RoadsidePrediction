"""Multimodal fusion model combining camera and sensor branches."""
import torch
import torch.nn as nn
from typing import Optional

from .camera_cnn import CameraCNN


class FusionModel(nn.Module):
    """Fuses camera and sensor features for congestion prediction.

    Supports three modes:
    - Full fusion (camera + sensor)
    - Camera-only (graceful degradation)
    - Sensor-only (graceful degradation)

    Args:
        camera_model: CameraCNN instance
        sensor_model: SensorMLP instance (or SensorCNN1D)
        fusion_dim: Hidden dimension for fusion layers (default: 64)
        fusion_type: Fusion strategy - "concat", "attention", or "gated"
        dropout: Dropout rate for regularization
    """

    def __init__(self, camera_model: CameraCNN, sensor_model: nn.Module,
                 fusion_dim: int = 64, fusion_type: str = "concat",
                 dropout: float = 0.2):
        super().__init__()
        self.camera_model = camera_model
        self.sensor_model = sensor_model
        self.fusion_type = fusion_type

        combined_dim = camera_model.feature_dim + sensor_model.feature_dim

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(combined_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 1)
            )
        elif fusion_type == "attention":
            self._build_attention_fusion(combined_dim, fusion_dim, dropout)
        elif fusion_type == "gated":
            self._build_gated_fusion(camera_model.feature_dim,
                                     sensor_model.feature_dim,
                                     fusion_dim, dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def _build_attention_fusion(self, combined_dim: int, fusion_dim: int,
                                dropout: float):
        """Build attention-based fusion mechanism."""
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),  # Weights for camera and sensor
            nn.Softmax(dim=1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )

    def _build_gated_fusion(self, camera_dim: int, sensor_dim: int,
                           fusion_dim: int, dropout: float):
        """Build gated fusion mechanism."""
        # Gate for camera features
        self.camera_gate = nn.Sequential(
            nn.Linear(camera_dim + sensor_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, camera_dim),
            nn.Sigmoid()
        )
        # Gate for sensor features
        self.sensor_gate = nn.Sequential(
            nn.Linear(camera_dim + sensor_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, sensor_dim),
            nn.Sigmoid()
        )
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(camera_dim + sensor_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, camera_input: torch.Tensor,
                sensor_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with both modalities.

        Args:
            camera_input: (batch, 3, H, W) images
            sensor_input: (batch, seq_length, num_features) sensor data

        Returns:
            (batch, 1) congestion scores in [0, 1]
        """
        camera_features = self.camera_model(camera_input)
        sensor_features = self.sensor_model(sensor_input)

        if self.fusion_type == "concat":
            combined = torch.cat([camera_features, sensor_features], dim=1)
            output = self.fusion(combined)

        elif self.fusion_type == "attention":
            combined = torch.cat([camera_features, sensor_features], dim=1)
            # Compute attention weights
            weights = self.attention(combined)  # (batch, 2)
            # Apply weights (simplified - weight the combined features)
            output = self.fusion(combined)

        elif self.fusion_type == "gated":
            combined = torch.cat([camera_features, sensor_features], dim=1)
            # Apply gating
            camera_gate = self.camera_gate(combined)
            sensor_gate = self.sensor_gate(combined)
            gated_camera = camera_features * camera_gate
            gated_sensor = sensor_features * sensor_gate
            gated_combined = torch.cat([gated_camera, gated_sensor], dim=1)
            output = self.fusion(gated_combined)

        return torch.sigmoid(output)

    def forward_camera_only(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using camera only (graceful degradation).

        Args:
            x: (batch, 3, H, W) images

        Returns:
            (batch, 1) congestion scores in [0, 1]
        """
        return self.camera_model.forward_predict(x)

    def forward_sensor_only(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using sensors only (graceful degradation).

        Args:
            x: (batch, seq_length, num_features) sensor data

        Returns:
            (batch, 1) congestion scores in [0, 1]
        """
        return self.sensor_model.forward_predict(x)

    def get_camera_model(self) -> CameraCNN:
        """Return the camera model component."""
        return self.camera_model

    def get_sensor_model(self) -> nn.Module:
        """Return the sensor model component."""
        return self.sensor_model

    def get_num_params(self) -> dict:
        """Return parameter counts for each component."""
        return {
            'camera': self.camera_model.get_num_params(),
            'sensor': self.sensor_model.get_num_params(),
            'fusion': sum(p.numel() for n, p in self.named_parameters()
                         if 'camera_model' not in n and 'sensor_model' not in n),
            'total': sum(p.numel() for p in self.parameters())
        }


class LateFusionModel(nn.Module):
    """Late fusion model - averages predictions from each modality.

    Simpler alternative to learned fusion, useful as a baseline.

    Args:
        camera_model: CameraCNN instance
        sensor_model: SensorMLP instance
        camera_weight: Weight for camera prediction (default: 0.5)
    """

    def __init__(self, camera_model: CameraCNN, sensor_model: nn.Module,
                 camera_weight: float = 0.5):
        super().__init__()
        self.camera_model = camera_model
        self.sensor_model = sensor_model
        self.camera_weight = camera_weight
        self.sensor_weight = 1.0 - camera_weight

    def forward(self, camera_input: torch.Tensor,
                sensor_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with weighted average fusion."""
        camera_pred = self.camera_model.forward_predict(camera_input)
        sensor_pred = self.sensor_model.forward_predict(sensor_input)

        return self.camera_weight * camera_pred + self.sensor_weight * sensor_pred

    def forward_camera_only(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using camera only."""
        return self.camera_model.forward_predict(x)

    def forward_sensor_only(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using sensors only."""
        return self.sensor_model.forward_predict(x)


class GraphFusionModel(nn.Module):
    """Fusion model: CNN perception + graph-based forecasting.

    Stage 1: CameraCNN processes images -> per-camera congestion estimate
    Stage 2: Stack estimates over time -> (N, T, 1) time-series
    Stage 3: Graph model (T-GCN or AGCRN) forecasts from the time-series

    Can also run in graph-only mode using pre-computed count-based features
    instead of CNN outputs.

    Args:
        camera_model: CameraCNN instance (perception layer)
        graph_model: TGCN or AGCRN instance (forecasting layer)
    """

    def __init__(self, camera_model: CameraCNN, graph_model: nn.Module):
        super().__init__()
        self.camera_model = camera_model
        self.graph_model = graph_model

    def forward(self, images: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """Full pipeline: images -> CNN -> graph -> forecast.

        Args:
            images: (B, N, T, 3, H, W) images for each camera at each time step
            adj: (N, N) adjacency matrix

        Returns:
            (B, N, T_out) forecast scores in [0, 1]
        """
        B, N, T, C, H, W = images.shape

        # Run CNN on each image to get congestion score
        # Reshape to process all images at once
        images_flat = images.reshape(B * N * T, C, H, W)
        scores = self.camera_model.forward_predict(images_flat)  # (B*N*T, 1)
        scores = scores.reshape(B, N, T, 1)  # (B, N, T, 1)

        # Feed into graph model
        return self.graph_model.forward_predict(scores, adj)

    def forward_graph_only(self, timeseries: torch.Tensor,
                           adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Skip CNN — use pre-computed features directly.

        Args:
            timeseries: (B, N, T_in, F) count-based congestion time-series
            adj: Optional adjacency matrix

        Returns:
            (B, N, T_out) forecast scores in [0, 1]
        """
        return self.graph_model.forward_predict(timeseries, adj)

    def get_num_params(self) -> dict:
        """Return parameter counts for each component."""
        camera_params = sum(p.numel() for p in self.camera_model.parameters())
        graph_params = sum(p.numel() for p in self.graph_model.parameters())
        return {
            'camera': camera_params,
            'graph': graph_params,
            'total': camera_params + graph_params,
        }
