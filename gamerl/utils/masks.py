"""
Causal mask generation for autoregressive Transformer.

Replaces the deprecated Variable-based nopeak_mask in Batch.py.
Uses modern PyTorch APIs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (upper-triangular) attention mask.

    Args:
        seq_len: Sequence length.
        device: Torch device.

    Returns:
        Boolean mask of shape (1, seq_len, seq_len).
        True where attention is allowed.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0)


def create_padding_mask(
    seq: torch.Tensor, pad_token: int = -1
) -> torch.Tensor:
    """
    Create a padding mask from a sequence.

    Args:
        seq: Token sequence of shape (batch, seq_len).
        pad_token: Token value representing padding.

    Returns:
        Boolean mask of shape (batch, 1, seq_len).
        True where tokens are NOT padding.
    """
    return (seq != pad_token).unsqueeze(-2)


def create_masks(
    src: torch.Tensor,
    trg: torch.Tensor,
    device: torch.device,
    pad_token: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create source and target masks for the Transformer.

    The target mask combines padding mask and causal mask.

    Args:
        src: Source sequence (batch, seq_len).
        trg: Target sequence (batch, seq_len).
        device: Torch device.
        pad_token: Padding token value.

    Returns:
        Tuple of (src_mask, trg_mask).
        - src_mask: (batch, 1, seq_len) - padding mask
        - trg_mask: (batch, seq_len, seq_len) - padding & causal mask
    """
    src_mask = create_padding_mask(src, pad_token).to(device)

    if trg is not None:
        trg_padding = create_padding_mask(trg, pad_token).to(device)  # (batch, 1, seq_len)
        causal = create_causal_mask(trg.size(1), device)  # (1, seq_len, seq_len)
        # Broadcast: (batch, 1, seq_len) & (1, seq_len, seq_len) -> (batch, seq_len, seq_len)
        trg_mask = trg_padding & causal
    else:
        trg_mask = None

    return src_mask, trg_mask
