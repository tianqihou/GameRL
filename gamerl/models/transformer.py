"""
Transformer-based policy network for action prediction.

Modernized version of the original hand-rolled Transformer in 模型_策略梯度.py.
Uses PyTorch 2.x native modules with:
- Pre-LayerNorm architecture (more stable training)
- RoPE positional encoding (better length generalization)
- GELU activation
- Proper weight initialization
- Separate policy and value heads
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    More effective than learned positional embeddings for sequence modeling,
    especially for variable-length game state sequences.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        self.d_model = d_model

        # Precompute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute position-frequency products
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, d_model/2)
        # Duplicate to full dimension
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, d_model)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            seq_len: Override sequence length.

        Returns:
            Tensor with rotary positional encoding applied.
        """
        if seq_len is None:
            seq_len = x.size(1)

        cos = self.cos_cached[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
        sin = self.sin_cached[:seq_len].unsqueeze(0)

        # Split into two halves for rotation
        x1, x2 = x.chunk(2, dim=-1)
        # Rotate: [x1, x2] * [cos, sin] with sign flip
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + rotated * sin


class TransformerPolicy(nn.Module):
    """
    Transformer-based policy-value network for game AI.

    Takes image features (from backbone) and action history, and outputs:
    - Action logits (policy head, discrete)
    - State value estimate (value head)
    - Continuous action mean (continuous head, optional)

    When ``continuous_dim > 0`` the network also outputs a mean vector
    for the continuous parameters.  The log-std is a learned parameter
    (not state-dependent), following common practice in PPO implementations.

    Architecture:
        Image features -> Linear projection -> + RoPE -> Transformer Decoder -> Heads

    Args:
        feature_dim: Input dimension of image features (grid*grid * channels).
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        vocab_size: Action vocabulary size.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length for RoPE.
        continuous_dim: Number of continuous action parameters (0 for pure discrete).
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        vocab_size: int = 131,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        continuous_dim: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.continuous_dim = continuous_dim

        # Project image features to transformer dimension
        self.feature_proj = nn.Linear(feature_dim, d_model)

        # Action embedding
        self.action_embed = nn.Embedding(vocab_size, d_model, padding_idx=-1)

        # RoPE positional encoding
        self.rope = RotaryPositionalEmbedding(d_model, max_seq_len)

        # Pre-LayerNorm Transformer encoder (self-attention only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Policy head (action logits)
        self.policy_head = nn.Linear(d_model, vocab_size)

        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Continuous action head (mean of Gaussian) — only when needed
        self.continuous_head: nn.Linear | None = None
        self.continuous_log_std: nn.Parameter | None = None
        if continuous_dim > 0:
            self.continuous_head = nn.Linear(d_model, continuous_dim)
            # Learnable log-std (state-independent), initialized near 0
            self.continuous_log_std = nn.Parameter(torch.zeros(continuous_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/GPT-style initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        image_features: torch.Tensor,
        action_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the policy-value network.

        Args:
            image_features: Image features from backbone, shape (batch, seq_len, feature_dim).
            action_seq: Action history tokens, shape (batch, seq_len).
            attn_mask: Causal attention mask, shape (seq_len, seq_len).
            key_padding_mask: Padding mask, shape (batch, seq_len).
                True values are positions to MASK OUT (PyTorch convention).

        Returns:
            Tuple of (action_logits, state_values, continuous_mean):
            - action_logits: (batch, seq_len, vocab_size)
            - state_values: (batch, seq_len, 1)
            - continuous_mean: (batch, seq_len, continuous_dim) or None
        """
        batch_size, seq_len, _ = image_features.shape

        # Project image features
        x = self.feature_proj(image_features)  # (B, S, d_model)

        # Add action embedding (blended with image features)
        # Use action embedding as a bias, matching original design intent
        action_emb = self.action_embed(action_seq.clamp(min=0))  # (B, S, d_model)
        x = x + action_emb * 0.1

        # Apply RoPE positional encoding
        x = self.rope(x, seq_len)

        # Convert boolean mask to float mask for PyTorch compatibility
        # PyTorch TransformerEncoder expects: float mask (0=attend, -inf=mask)
        # or boolean mask (True=attend). We use float for maximum compatibility.
        float_mask = None
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
            float_mask.masked_fill_(~attn_mask, float("-inf"))
        else:
            float_mask = attn_mask

        # Transformer (self-attention with causal mask)
        x = self.transformer(
            x,
            mask=float_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=(float_mask is not None),
        )

        x = self.final_norm(x)

        # Heads
        action_logits = self.policy_head(x)  # (B, S, vocab_size)
        state_values = self.value_head(x)    # (B, S, 1)

        # Continuous action mean (optional)
        continuous_mean = None
        if self.continuous_head is not None:
            continuous_mean = self.continuous_head(x)  # (B, S, continuous_dim)
            # Tanh squashes to [-1, 1] to match the parameter range
            continuous_mean = torch.tanh(continuous_mean)

        return action_logits, state_values, continuous_mean

    def get_last_step(
        self,
        image_features: torch.Tensor,
        action_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get policy and value for only the last timestep (for inference).

        This is more efficient than computing the full sequence when
        we only need the next action.

        Returns:
            Tuple of (action_logits_last, value_last, continuous_mean_last):
            - action_logits_last: (batch, vocab_size)
            - value_last: (batch, 1)
            - continuous_mean_last: (batch, continuous_dim) or None
        """
        logits, values, cont_mean = self.forward(image_features, action_seq, attn_mask, key_padding_mask)
        cont_last = cont_mean[:, -1, :] if cont_mean is not None else None
        return logits[:, -1, :], values[:, -1, :], cont_last

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
