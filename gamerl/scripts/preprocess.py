#!/usr/bin/env python3
"""
Preprocess collected training data.

Usage:
    python -m gamerl.scripts.preprocess --data ../training_data
"""

from __future__ import annotations

import argparse
import logging

from ..config import Config
from ..data.preprocessor import DataPreprocessor
from ..models.backbone import BackboneExtractor
from ..utils.actions import ActionSpace
from ..utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Preprocess WZCQ training data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data", default="../training_data", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    args = parser.parse_args()

    setup_logger("gamerl", log_file="logs/preprocess.log")

    config = Config.from_yaml(args.config)

    backbone = BackboneExtractor(
        backbone_name=config.model.backbone,
        grid_size=config.model.backbone_grid_size,
        pretrained=config.model.pretrained,
        freeze=True,
        use_half=config.model.backbone_half,
    )

    action_space = ActionSpace()
    preprocessor = DataPreprocessor(backbone, action_space, args.batch_size)

    results = preprocessor.process_directory(args.data)
    print(f"Processed {len(results)} episodes")


if __name__ == "__main__":
    main()
