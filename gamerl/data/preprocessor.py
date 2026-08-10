"""
Data preprocessor for offline training.

Processes collected screenshots and action records into precomputed
feature arrays for efficient training.

Replaces 处理训练数据5.py from the original project, with:
- Batch processing (instead of one image at a time)
- Memory-efficient feature storage
- Configurable backbone
- Universal action space support (continuous params preserved)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..models.backbone import BackboneExtractor
from ..utils.actions import BOS_TOKEN, UniversalActionSpace

logger = logging.getLogger("gamerl.data")


class DataPreprocessor:
    """
    Preprocess collected gameplay data into feature arrays.

    Takes raw screenshots + action JSONL and produces .npz files
    containing pre-extracted backbone features and action sequences.

    Handles both record formats:
    - **Universal**: ``touch_type`` + ``continuous_params`` per record
      (continuous sequence is saved as ``continuous_params`` in the npz).
    - **Legacy**: ``action_token`` per record (discrete only).

    Args:
        backbone: Image feature extraction backbone.
        bos_token: Beginning-of-sequence token.  Defaults to the
            universal BOS (WAIT=6); pass ``BOS_TOKEN`` (128) explicitly
            when preprocessing legacy per-game data.
        batch_size: Batch size for image processing.
        device: Device for feature extraction.
    """

    def __init__(
        self,
        backbone: BackboneExtractor,
        bos_token: int = UniversalActionSpace.BOS_TOKEN,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.backbone = backbone
        self.bos_token = bos_token
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_profile(
        cls,
        backbone: BackboneExtractor,
        profile,
        batch_size: int = 32,
        device: str = "cuda",
    ) -> "DataPreprocessor":
        """Create a preprocessor configured for a game profile's action mode."""
        return cls(
            backbone=backbone,
            bos_token=profile.bos_token,
            batch_size=batch_size,
            device=device,
        )

    def process_episode(
        self, episode_dir: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Process a single episode directory.

        Args:
            episode_dir: Directory containing .jpg screenshots and actions.jsonl.

        Returns:
            Tuple of (image_features, action_sequence, continuous_sequence):
            - image_features: (seq_len, feature_dim) float32 array
            - action_sequence: (seq_len + 1,) int64 array (BOS prepended)
            - continuous_sequence: (seq_len + 1, 5) float32 array, or None
              when the records contain no continuous params (legacy data)
        """
        episode_dir = Path(episode_dir)
        actions_path = episode_dir / "actions.jsonl"

        empty = (np.array([]), np.array([]), None)

        if not actions_path.exists():
            logger.warning(f"No actions.jsonl in {episode_dir}")
            return empty

        # Load action records
        records = []
        with open(actions_path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        if not records:
            return empty

        # Build action sequence (prepend BOS token).
        # Universal records carry "touch_type"; legacy records carry "action_token".
        action_seq = [self.bos_token]
        for rec in records:
            if "touch_type" in rec:
                action_seq.append(int(rec["touch_type"]))
            else:
                action_seq.append(int(rec["action_token"]))

        # Build continuous param sequence when present (universal mode).
        # BOS row gets neutral params (screen center, no movement).
        has_continuous = any("continuous_params" in rec for rec in records)
        continuous_seq: Optional[np.ndarray] = None
        if has_continuous:
            neutral = UniversalActionSpace.neutral_params().tolist()
            rows = [neutral]
            for rec in records:
                rows.append(rec.get("continuous_params", neutral))
            continuous_seq = np.array(rows, dtype=np.float32)

        # Extract features in batches
        all_features = []
        images = []
        for rec in records:
            img_path = episode_dir / rec["image"]
            if img_path.exists():
                img = Image.open(img_path)
                arr = np.array(img)
                images.append(arr)

        # Process in batches
        for i in tqdm(range(0, len(images), self.batch_size), desc="Extracting features"):
            batch_images = images[i:i + self.batch_size]
            batch_tensor = torch.stack([
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                for img in batch_images
            ]).to(self.device)

            with torch.no_grad():
                features = self.backbone(batch_tensor)  # (B, grid*grid, C)
                # Flatten: (B, grid*grid * C)
                features_flat = features.reshape(features.size(0), -1)
                all_features.append(features_flat.cpu().numpy())

        image_features = np.concatenate(all_features, axis=0)  # (seq_len, feature_dim)
        action_sequence = np.array(action_seq, dtype=np.int64)

        return image_features, action_sequence, continuous_seq

    def process_and_save(self, episode_dir: str | Path) -> Path:
        """
        Process an episode and save the preprocessed data.

        Args:
            episode_dir: Directory containing raw data.

        Returns:
            Path to the saved .npz file.
        """
        episode_dir = Path(episode_dir)
        output_path = episode_dir / "preprocessed.npz"

        image_features, action_sequence, continuous_seq = self.process_episode(episode_dir)

        if len(image_features) == 0:
            logger.warning(f"No data to save for {episode_dir}")
            return output_path

        arrays = {
            "image_features": image_features,
            "action_sequence": action_sequence,
        }
        if continuous_seq is not None:
            arrays["continuous_params"] = continuous_seq

        np.savez_compressed(output_path, **arrays)

        logger.info(
            f"Saved preprocessed data: {output_path} "
            f"({image_features.shape[0]} frames, {image_features.shape[1]} dims"
            f"{', +continuous' if continuous_seq is not None else ''})"
        )

        return output_path

    def process_directory(self, data_dir: str | Path) -> List[Path]:
        """
        Process all episodes in a directory.

        Args:
            data_dir: Directory containing episode subdirectories.

        Returns:
            List of paths to saved .npz files.
        """
        data_dir = Path(data_dir)
        results = []

        for episode_dir in sorted(data_dir.iterdir()):
            if not episode_dir.is_dir():
                continue

            npz_path = episode_dir / "preprocessed.npz"
            if npz_path.exists():
                logger.info(f"Skipping {episode_dir} (already preprocessed)")
                continue

            try:
                path = self.process_and_save(episode_dir)
                results.append(path)
            except Exception as e:
                logger.error(f"Failed to process {episode_dir}: {e}")

        return results
