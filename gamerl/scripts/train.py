#!/usr/bin/env python3
"""
Train the policy network.

Usage:
    # Supervised pretraining
    python -m gamerl.scripts.train --mode supervised --data ../training_data

    # PPO fine-tuning
    python -m gamerl.scripts.train --mode ppo --weights weights/policy_latest.pt
"""

from __future__ import annotations

import argparse
import logging

from ..config import Config
from ..training.trainer import PolicyTrainer
from ..utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train GameRL policy network")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["supervised", "ppo"], required=True,
                        help="Training mode")
    parser.add_argument("--data", default=None,
                        help="Training data directory (default: imitation.dataset_path from config)")
    parser.add_argument("--weights", default=None, help="Weights to resume/load")
    parser.add_argument("--state-model", default=None, help="State model weights for PPO rewards")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    setup_logger("gamerl", log_file="logs/train.log")
    logger = logging.getLogger("gamerl")

    config = Config.from_yaml(args.config)
    trainer = PolicyTrainer(config)

    if args.weights:
        trainer.resume(args.weights)

    if args.mode == "supervised":
        trainer.train_supervised(data_dir=args.data, epochs=args.epochs)

    elif args.mode == "ppo":
        if args.state_model:
            trainer.load_state_model(args.state_model)

        # For PPO, we need a live environment.
        # GameEnvironment auto-detects universal vs legacy mode from the profile.
        from ..environment.capture import ScreenCapture
        from ..environment.device import ADBDevice
        from ..environment.game_env import GameEnvironment

        device = ADBDevice(serial=config.device.serial)
        capture = ScreenCapture(
            method=config.device.capture_method,
            window_title=config.device.window_title,
            target_size=tuple(config.device.screenshot_size),
            crop_box=config.device.crop_box,
        )
        env = GameEnvironment(capture, device, trainer.backbone, profile=trainer.profile)

        trainer.train_ppo(env=env, episodes=args.epochs or 100)


if __name__ == "__main__":
    main()
