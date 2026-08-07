"""
State judgment model for game event classification.

Classifies game screenshots into event categories. The specific categories
depend on the game profile's state_classes property:

- Honor of Kings: kill_minion, kill_tower, kill_hero, assist_kill,
  attacked_by_tower, killed, death, normal
- Peacekeeper: parachuting, looting, combat, driving, final_circle
- Genshin: overworld, combat, menu, dialogue

This is used for reward shaping during training via the RewardShaper.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .transformer import TransformerPolicy


class StateJudgmentModel(nn.Module):
    """
    Event classifier based on a small Transformer.

    Uses the same architecture as the policy network but with fewer layers
    and a classification head instead of policy/value heads.

    Args:
        feature_dim: Input dimension of image features.
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer layers (small, e.g. 2).
        n_heads: Number of attention heads.
        num_classes: Number of event classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 768,
        n_layers: int = 2,
        n_heads: int = 12,
        num_classes: int = 6,
        vocab_size: int = 130,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Reuse the transformer architecture but replace the heads
        self.backbone_proj = nn.Linear(feature_dim, d_model)
        self.action_embed = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image_features: torch.Tensor,
        action_seq: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Classify game events from image features.

        Args:
            image_features: (batch, seq_len, feature_dim)
            action_seq: (batch, seq_len) - action tokens (can be all ones for single-frame)
            attn_mask: Causal attention mask.
            key_padding_mask: Padding mask.

        Returns:
            Class logits of shape (batch, seq_len, num_classes).
        """
        x = self.backbone_proj(image_features)
        action_emb = self.action_embed(action_seq.clamp(min=0))
        x = x + action_emb * 0.1

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.classifier(x)

        return logits

    def classify_single_frame(
        self,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify a single frame (seq_len=1).

        Args:
            image_features: (batch, feature_dim) or (batch, 1, feature_dim)

        Returns:
            Class logits of shape (batch, num_classes).
        """
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)

        batch_size = image_features.size(0)
        action_seq = torch.ones(batch_size, 1, dtype=torch.long, device=image_features.device)

        logits = self.forward(image_features, action_seq)
        return logits.squeeze(1)  # (batch, num_classes)
