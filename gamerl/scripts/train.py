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

        # Vision pipeline (structured state fused into the policy)
        detector, state_builder = None, None
        if config.vision.enabled:
            from ..vision.detector import GameDetector, MockDetector
            from ..vision.state_builder import GameStateBuilder

            if config.vision.detector_backend == "yolo" and config.vision.model_path:
                detector = GameDetector(
                    model_path=config.vision.model_path,
                    class_names=trainer.profile.detection_classes,
                    conf_threshold=config.vision.conf_threshold,
                    iou_threshold=config.vision.iou_threshold,
                    input_size=tuple(config.vision.input_size),
                )
            else:
                detector = MockDetector(class_names=trainer.profile.detection_classes)
            state_builder = GameStateBuilder(
                class_names=trainer.profile.detection_classes,
                screen_resolution=trainer.profile.resolution,
                max_enemies=config.vision.max_enemies,
                max_towers=config.vision.max_towers,
                max_minions=config.vision.max_minions,
                num_skills=trainer.profile.num_skills,
            )

        env = GameEnvironment(
            capture, device, trainer.backbone, profile=trainer.profile,
            detector=detector, state_builder=state_builder,
        )

        trainer.train_ppo(env=env, episodes=args.epochs or 100)


if __name__ == "__main__":
    main()
