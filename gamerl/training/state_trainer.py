"""
State judgment model trainer.

Trains the event classifier that provides reward signals
for the PPO policy network.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..models.backbone import BackboneExtractor
from ..models.state_judgment import StateJudgmentModel
from ..utils.logging import MetricsLogger

logger = logging.getLogger("gamerl.training")


class LabeledEventDataset(Dataset):
    """Dataset of labeled game event screenshots."""

    def __init__(
        self,
        data_dir: str | Path,
        backbone: BackboneExtractor,
        state_classes: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        self.data_dir = Path(data_dir)
        self.backbone = backbone
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build event → index mapping from game profile's state_classes
        if state_classes is None:
            # Default to Honor of Kings classes for backward compatibility
            state_classes = [
                "kill_minion", "kill_tower", "kill_hero", "assist_kill",
                "attacked_by_tower", "killed", "death", "normal",
            ]
        self.event_classes = {name: idx for idx, name in enumerate(state_classes)}

        self.samples: List[tuple[str, int]] = []
        self._load_labels()

    def _load_labels(self) -> None:
        """Load labeled events from JSON file."""
        label_file = self.data_dir / "labels.jsonl"

        if not label_file.exists():
            logger.warning(f"No label file found at {label_file}")
            return

        with open(label_file, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for key, label in record.items():
                    if label in self.event_classes:
                        img_path = self.data_dir / f"{key}.jpg"
                        if img_path.exists():
                            self.samples.append((str(img_path), self.event_classes[label]))

        logger.info(f"Loaded {len(self.samples)} labeled samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        arr = np.array(img)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        return {
            "image": tensor,
            "label": label,
        }


class StateModelTrainer:
    """
    Trainer for the state judgment (event classification) model.

    Args:
        config: Top-level configuration.
        device: Torch device.
    """

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.backbone = BackboneExtractor(
            backbone_name=config.model.backbone,
            grid_size=config.model.backbone_grid_size,
            pretrained=config.model.pretrained,
            freeze=True,
            use_half=config.model.backbone_half,
        ).to(self.device)

        feature_dim = self.backbone.get_flat_dim()

        # Resolve state classes from the game profile
        from ..profiles import get_profile
        try:
            profile = get_profile(config.game.name)
            self.state_classes = profile.state_classes
        except (ValueError, KeyError):
            self.state_classes = None  # Will use default in dataset

        self.model = StateJudgmentModel(
            feature_dim=feature_dim,
            d_model=config.state_model.d_model,
            n_layers=config.state_model.n_layers,
            n_heads=config.state_model.n_heads,
            num_classes=len(self.state_classes) if self.state_classes else config.state_model.num_classes,
            dropout=config.state_model.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=6.25e-5,
            betas=(0.9, 0.98),
            weight_decay=0.01,
            eps=1e-9,
        )

        self.metrics = MetricsLogger(config.training.log_dir)

    def train(
        self,
        data_dir: str,
        epochs: int = 100,
        batch_size: int = 1,
    ) -> None:
        """
        Train the state judgment model.

        Args:
            data_dir: Directory containing labeled event data.
            epochs: Number of training epochs.
            batch_size: Batch size (typically 1 due to variable image sizes).
        """
        dataset = LabeledEventDataset(
            data_dir, self.backbone,
            state_classes=self.state_classes,
            device=str(self.device),
        )

        if len(dataset) == 0:
            logger.error("No training data found!")
            return

        weights_dir = Path(self.config.training.weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0

        for epoch in range(epochs):
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            epoch_loss = 0.0
            correct = 0
            total = 0

            for idx in indices:
                sample = dataset[idx]
                image = sample["image"].unsqueeze(0).to(self.device)
                label = torch.tensor([sample["label"]], device=self.device)

                # Extract features
                with torch.no_grad():
                    features = self.backbone(image)  # (1, grid*grid, C)
                    features_flat = features.reshape(1, -1)  # (1, grid*grid * C)

                # Forward
                logits = self.model.classify_single_frame(features_flat)

                loss = F.cross_entropy(logits, label)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                pred = logits.argmax(dim=-1)
                correct += (pred == label).sum().item()
                total += 1
                global_step += 1

            avg_loss = epoch_loss / max(total, 1)
            accuracy = correct / max(total, 1)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}"
            )

            self.metrics.log_scalar("state_model/loss", avg_loss, epoch)
            self.metrics.log_scalar("state_model/accuracy", accuracy, epoch)

            # Save checkpoint
            torch.save(
                self.model.state_dict(),
                weights_dir / f"state_model_epoch{epoch + 1}.pt",
            )
            torch.save(
                self.model.state_dict(),
                weights_dir / "state_model_latest.pt",
            )

        self.metrics.close()
        logger.info("State model training complete.")
