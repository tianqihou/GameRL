#!/usr/bin/env python3
"""
Train the state judgment (event classification) model.

Usage:
    python -m gamerl.scripts.train_state_model --data ../labeled_data
"""

from __future__ import annotations

import argparse
import logging

from ..config import Config
from ..training.state_trainer import StateModelTrainer
from ..utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train WZCQ state judgment model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data", required=True, help="Labeled data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    args = parser.parse_args()

    setup_logger("gamerl", log_file="logs/train_state.log")

    config = Config.from_yaml(args.config)
    trainer = StateModelTrainer(config)
    trainer.train(data_dir=args.data, epochs=args.epochs)


if __name__ == "__main__":
    main()
