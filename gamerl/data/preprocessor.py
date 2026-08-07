"""
Data preprocessor for offline training.

Processes collected screenshots and action records into precomputed
feature arrays for efficient training.

Replaces 处理训练数据5.py from the original project, with:
- Batch processing (instead of one image at a time)
- Memory-efficient feature storage
- Configurable backbone
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..models.backbone import BackboneExtractor
from ..utils.actions import ActionSpace, BOS_TOKEN

logger = logging.getLogger("gamerl.data")


class DataPreprocessor:
    """
    Preprocess collected gameplay data into feature arrays.

    Takes raw screenshots + action JSONL and produces .npz files
    containing pre-extracted backbone features and action sequences.

    Args:
        backbone: Image feature extraction backbone.
        action_space: Action space for encoding actions.
        batch_size: Batch size for image processing.
    """

    def __init__(
        self,
        backbone: BackboneExtractor,
        action_space: ActionSpace,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.backbone = backbone
        self.action_space = action_space
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def process_episode(self, episode_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single episode directory.

        Args:
            episode_dir: Directory containing .jpg screenshots and actions.jsonl.

        Returns:
            Tuple of (image_features, action_sequence):
            - image_features: (seq_len, feature_dim) float32 array
            - action_sequence: (seq_len,) int64 array
        """
        episode_dir = Path(episode_dir)
        actions_path = episode_dir / "actions.jsonl"

        if not actions_path.exists():
            logger.warning(f"No actions.jsonl in {episode_dir}")
            return np.array([]), np.array([])

        # Load action records
        records = []
        with open(actions_path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        if not records:
            return np.array([]), np.array([])

        # Build action sequence (prepend BOS token)
        action_seq = [BOS_TOKEN]
        for rec in records:
            action_seq.append(rec["action_token"])

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

        return image_features, action_sequence

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

        image_features, action_sequence = self.process_episode(episode_dir)

        if len(image_features) == 0:
            logger.warning(f"No data to save for {episode_dir}")
            return output_path

        np.savez_compressed(
            output_path,
            image_features=image_features,
            action_sequence=action_sequence,
        )

        logger.info(
            f"Saved preprocessed data: {output_path} "
            f"({image_features.shape[0]} frames, {image_features.shape[1]} dims)"
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
