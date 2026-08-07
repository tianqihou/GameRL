"""
PyTorch Dataset for loading preprocessed training data.

Provides efficient loading of preprocessed .npz files with
configurable sequence chunking for variable-length episodes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("gamerl.data")


class GameSequenceDataset(Dataset):
    """
    Dataset of game state sequences for training.

    Loads preprocessed .npz files and chunks them into
    fixed-length sequences for Transformer training.

    Args:
        data_dir: Directory containing preprocessed .npz files.
        chunk_size: Length of each sequence chunk.
        stride: Stride between consecutive chunks (for overlap).
    """

    def __init__(
        self,
        data_dir: str | Path,
        chunk_size: int = 600,
        stride: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.stride = stride or chunk_size

        # Load all preprocessed data
        self._episodes: List[Tuple[np.ndarray, np.ndarray]] = []
        self._chunk_indices: List[Tuple[int, int, int]] = []  # (episode_idx, start, end)

        self._load_data()

    def _load_data(self) -> None:
        """Load all .npz files and build chunk index."""
        for npz_path in sorted(self.data_dir.rglob("preprocessed.npz")):
            data = np.load(npz_path, allow_pickle=True)
            image_features = data["image_features"]  # (seq_len, feature_dim)
            action_sequence = data["action_sequence"]  # (seq_len,)

            if len(image_features) < 2:
                continue

            ep_idx = len(self._episodes)
            self._episodes.append((image_features, action_sequence))

            # Build chunk indices
            seq_len = len(action_sequence)
            for start in range(0, seq_len - 1, self.stride):
                end = min(start + self.chunk_size, seq_len)
                if end - start >= 2:  # Need at least 2 timesteps
                    self._chunk_indices.append((ep_idx, start, end))
                if end >= seq_len:
                    break

        logger.info(
            f"Loaded {len(self._episodes)} episodes, "
            f"{len(self._chunk_indices)} chunks "
            f"(chunk_size={self.chunk_size}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self._chunk_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sequence chunk.

        Returns:
            Dictionary with:
            - "image_features": (seq_len, feature_dim)
            - "actions": (seq_len,) - input actions
            - "target_actions": (seq_len,) - shifted by 1 (next action prediction)
        """
        ep_idx, start, end = self._chunk_indices[idx]
        image_features, action_sequence = self._episodes[ep_idx]

        chunk_features = image_features[start:end]
        chunk_actions = action_sequence[start:end]
        chunk_targets = action_sequence[start + 1:end + 1] if end < len(action_sequence) else \
            np.append(action_sequence[start + 1:end], 0)

        return {
            "image_features": torch.FloatTensor(chunk_features),
            "actions": torch.LongTensor(chunk_actions),
            "target_actions": torch.LongTensor(chunk_targets),
        }


def collate_sequences(batch: List[dict]) -> dict:
    """
    Collate function for variable-length sequences.

    Pads sequences to the same length within a batch.

    Args:
        batch: List of dictionaries from __getitem__.

    Returns:
        Padded batch dictionary.
    """
    max_len = max(item["actions"].size(0) for item in batch)
    feature_dim = batch[0]["image_features"].size(-1)

    padded_features = torch.zeros(len(batch), max_len, feature_dim)
    padded_actions = torch.full((len(batch), max_len), -1, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_len), -1, dtype=torch.long)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)  # True = padding

    for i, item in enumerate(batch):
        seq_len = item["actions"].size(0)
        padded_features[i, :seq_len] = item["image_features"]
        padded_actions[i, :seq_len] = item["actions"]
        padded_targets[i, :seq_len] = item["target_actions"]
        padding_mask[i, :seq_len] = False  # False = not padding

    return {
        "image_features": padded_features,
        "actions": padded_actions,
        "target_actions": padded_targets,
        "padding_mask": padding_mask,  # True where padding (to be masked out)
    }
