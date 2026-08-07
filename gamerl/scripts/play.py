#!/usr/bin/env python3
"""
Run the trained AI to play the game.

Usage:
    python -m gamerl.scripts.play --weights weights/policy_latest.pt
"""

from __future__ import annotations

import argparse
import logging
import time

import torch

from ..config import Config
from ..environment.capture import ScreenCapture
from ..environment.device import ADBDevice, ActionMapper
from ..environment.game_env import GameEnvironment
from ..models.backbone import BackboneExtractor
from ..models.transformer import TransformerPolicy
from ..agent.ppo import PPOAgent
from ..utils.actions import ActionSpace
from ..utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run WZCQ AI to play")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--weights", required=True, help="Policy weights path")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max steps")
    parser.add_argument("--fps", type=int, default=5, help="Target FPS")
    args = parser.parse_args()

    setup_logger("gamerl", log_file="logs/play.log")
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

    # Build and load policy
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

    logger.info("Starting AI gameplay...")
    state = env.reset()
    frame_interval = 1.0 / args.fps

    for step in range(args.max_steps):
        loop_start = time.time()

        # Select action
        action, log_prob, value = agent.select_action(
            state.image_features,
            state.action_history,
        )

        # Execute
        state, reward, done, info = env.step(action)

        movement, action_type = action_space.decode(action)
        logger.info(f"Step {step}: {movement.value}_{action_type.value} (value={value:.3f})")

        # Auto-buy items
        if step > 0 and step % config.collection.auto_buy_interval == 0:
            for skill_name in ["加三技能", "加二技能", "加一技能", "购买"]:
                cmd = action_mapper.get_action_command(skill_name)
                if cmd:
                    device.send_touch_command(cmd)

        # Maintain FPS
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        if done:
            logger.info("Game ended. Resetting...")
            state = env.reset()

    env.close()
    logger.info("Gameplay ended.")


if __name__ == "__main__":
    main()
