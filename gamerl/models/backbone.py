"""
Modern image backbone for feature extraction.

Replaces the original ResNet101 with configurable modern backbones:
- ConvNeXt-Tiny (28M params, ~2x faster than ResNet101)
- EfficientNet-V2-S (21M params)
- ResNet50/101 (backward compatibility)

All backbones output a spatial feature map of shape (B, grid*grid, C)
that feeds into the Transformer policy network.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BackboneExtractor(nn.Module):
    """
    Extract spatial feature maps from game screenshots.

    Wraps a torchvision model and adapts its output to a
    (batch, grid*grid, channels) feature tensor suitable for
    the Transformer decoder.

    Args:
        backbone_name: One of "convnext_tiny", "efficientnet_v2_s",
                       "resnet50", "resnet101".
        grid_size: Spatial size of the output feature map (grid_size x grid_size).
        pretrained: Whether to use pretrained weights.
        freeze: Whether to freeze backbone parameters.
        use_half: Whether to use float16 for the backbone (inference speedup).
    """

    # Channel dimensions for each backbone's final feature map
    _CHANNEL_MAP = {
        "convnext_tiny": 768,
        "efficientnet_v2_s": 1280,
        "resnet50": 2048,
        "resnet101": 2048,
    }

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        grid_size: int = 6,
        pretrained: bool = True,
        freeze: bool = True,
        use_half: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.grid_size = grid_size
        self.use_half = use_half and pretrained
        self.out_channels = self._CHANNEL_MAP[backbone_name]

        # Load backbone
        weights = "DEFAULT" if pretrained else None
        if backbone_name == "convnext_tiny":
            base = torchvision.models.convnext_tiny(weights=weights)
            # ConvNeXt: use features up to the last stage (before classifier)
            self.features = base.features
        elif backbone_name == "efficientnet_v2_s":
            base = torchvision.models.efficientnet_v2_s(weights=weights)
            self.features = base.features
        elif backbone_name == "resnet50":
            base = torchvision.models.resnet50(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
        elif backbone_name == "resnet101":
            base = torchvision.models.resnet101(weights=weights)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()

        if self.use_half:
            self.features.half()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a batch of images.

        Args:
            img: Image tensor of shape (batch, 3, H, W), normalized to [0, 1].

        Returns:
            Feature tensor of shape (batch, grid*grid, out_channels).
        """
        # Ensure correct dtype for frozen half-precision backbone
        if self.use_half and img.dtype != torch.float16:
            img = img.half()

        with torch.no_grad():
            x = self.features(img)
            # Adaptive pool to desired grid size
            x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
            # Reshape: (B, C, H, W) -> (B, H*W, C)
            x = x.flatten(2).transpose(1, 2)

        if self.use_half:
            x = x.float()

        return x

    def get_output_dim(self) -> int:
        """Return the output channel dimension."""
        return self.out_channels

    def get_flat_dim(self) -> int:
        """Return the flattened output dimension (grid*grid * channels)."""
        return self.grid_size * self.grid_size * self.out_channels
