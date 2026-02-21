"""CNN model for camera-based congestion prediction."""
import torch
import torch.nn as nn
from typing import Optional

try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class CameraCNN(nn.Module):
    """Small CNN for camera clients in federated learning.

    Uses MobileNetV2 backbone (efficient for edge deployment) with custom head.
    Can also use a smaller custom architecture for resource-constrained settings.

    Args:
        feature_dim: Dimension of feature vector output for fusion (default: 128)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        backbone: Backbone architecture - "mobilenetv2" or "custom_small"
        dropout: Dropout rate for regularization
    """

    def __init__(self, feature_dim: int = 128, pretrained: bool = True,
                 backbone: str = "mobilenetv2", dropout: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone_name = backbone

        if backbone == "mobilenetv2" and TORCHVISION_AVAILABLE:
            self._build_mobilenet(pretrained, dropout)
        else:
            self._build_custom(dropout)

    def _build_mobilenet(self, pretrained: bool, dropout: float):
        """Build MobileNetV2-based architecture."""
        # Backbone: MobileNetV2 (efficient, ~3.4M params)
        weights = 'IMAGENET1K_V1' if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features  # Output: (batch, 1280, 7, 7) for 224x224 input
        self.backbone_out_dim = 1280

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature projection head
        self.fc_features = nn.Sequential(
            nn.Linear(self.backbone_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.feature_dim),
            nn.ReLU()
        )

        # Prediction head (for standalone use)
        self.fc_predict = nn.Linear(self.feature_dim, 1)

    def _build_custom(self, dropout: float):
        """Build smaller custom CNN architecture (~500K params)."""
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Block 5: 14x14 -> 7x7
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.backbone_out_dim = 256

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature projection head
        self.fc_features = nn.Sequential(
            nn.Linear(self.backbone_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.feature_dim),
            nn.ReLU()
        )

        # Prediction head (for standalone use)
        self.fc_predict = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for fusion layer.

        Args:
            x: (batch, 3, H, W) input images

        Returns:
            (batch, feature_dim) feature vectors
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_features(x)
        return x

    def forward_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict congestion score (standalone mode).

        Args:
            x: (batch, 3, H, W) input images

        Returns:
            (batch, 1) congestion scores in [0, 1]
        """
        features = self.forward(x)
        output = self.fc_predict(features)
        return torch.sigmoid(output)

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone weights."""
        for param in self.features.parameters():
            param.requires_grad = True
