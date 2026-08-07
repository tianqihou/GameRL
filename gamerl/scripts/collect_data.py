#!/usr/bin/env python3
"""
Collect training data by playing the game.

Usage:
    python -m gamerl.scripts.collect_data --config configs/default.yaml
    python -m gamerl.scripts.collect_data --manual  # Human-controlled
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..config import Config
from ..data.collector import DataCollector
from ..environment.capture import ScreenCapture
from ..environment.device import ADBDevice, ActionMapper
from ..environment.game_env import GameEnvironment
from ..models.backbone import BackboneExtractor
from ..models.transformer import TransformerPolicy
from ..agent.ppo import PPOAgent
from ..utils.actions import ActionSpace
from ..utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Collect WZCQ training data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--manual", action="store_true", help="Manual (human) control mode")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--weights", default=None, help="Policy weights to load for AI mode")
    args = parser.parse_args()

    setup_logger("gamerl", log_file="logs/collect.log")
    logger = logging.getLogger("gamerl")

    config = Config.from_yaml(args.config)

    # Build components
    action_space = ActionSpace()
    device = ADBDevice(serial=config.device.serial)
    capture = ScreenCapture(
        method=config.device.capture_method,
        window_title=config.device.window_title,
        target_size=tuple(config.device.screenshot_size),
        crop_box=config.device.crop_box,
    )
    backbone = BackboneExtractor(
        backbone_name=config.model.backbone,
        grid_size=config.model.backbone_grid_size,
        pretrained=config.model.pretrained,
        freeze=True,
        use_half=config.model.backbone_half,
    )
    action_mapper = ActionMapper(device.get_screen_resolution())

    env = GameEnvironment(capture, device, backbone, action_space, action_mapper)

    # Build agent (optional, for AI-driven collection)
    agent = None
    if not args.manual and args.weights:
        import torch
        feature_dim = backbone.get_flat_dim()
        policy = TransformerPolicy(
            feature_dim=feature_dim,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            vocab_size=action_space.vocab_size,
            dropout=config.model.dropout,
        )
        agent = PPOAgent(config.agent, policy, backbone=None, device="cuda")
        agent.load(args.weights)

    collector = DataCollector(env, agent, action_space, config.collection.save_dir)

    for ep in range(args.episodes):
        logger.info(f"=== Episode {ep + 1}/{args.episodes} ===")
        collector.collect_episode(
            max_steps=args.max_steps,
            auto_buy_interval=config.collection.auto_buy_interval,
            target_fps=config.collection.target_fps,
            manual_override=args.manual,
        )


if __name__ == "__main__":
    main()
