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

from ..agent.ppo import PPOAgent
from ..config import Config
from ..environment.capture import ScreenCapture
from ..environment.device import ADBDevice
from ..environment.game_env import GameEnvironment
from ..models.backbone import BackboneExtractor
from ..profiles import get_profile
from ..utils.actions import TouchType, UniversalActionSpace
from ..utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run GameRL AI to play")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--weights", required=True, help="Policy weights path")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max steps")
    parser.add_argument("--fps", type=int, default=5, help="Target FPS")
    args = parser.parse_args()

    setup_logger("gamerl", log_file="logs/play.log")
    logger = logging.getLogger("gamerl")

    config = Config.from_yaml(args.config)
    profile = get_profile(config.game.name)
    logger.info(
        f"Game: {profile.display_name} "
        f"(action_mode={profile.action_mode}, vocab={profile.vocab_size})"
    )

    # Build components
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
    # GameEnvironment auto-detects universal vs legacy mode from the profile
    env = GameEnvironment(capture, device, backbone, profile=profile)

    # Build and load policy (auto-configured for the profile's action mode)
    feature_dim = backbone.get_flat_dim()
    agent = PPOAgent.from_profile(
        profile,
        config.agent,
        backbone=None,
        device="cuda",
        feature_dim=feature_dim,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
    )
    agent.load(args.weights)

    logger.info("Starting AI gameplay...")
    state = env.reset()
    frame_interval = 1.0 / args.fps

    for step in range(args.max_steps):
        loop_start = time.time()

        # Select action (5-tuple: action, log_prob, value, cont_params, cont_log_prob)
        action, log_prob, value, cont_params, _ = agent.select_action(
            state.image_features,
            state.action_history,
        )

        # Convert continuous params array to dict for env.step
        params_dict = None
        if cont_params is not None:
            params_dict = {
                name: float(cont_params[i])
                for i, name in enumerate(profile.continuous_params)
            }

        # Execute
        state, reward, done, info = env.step(action, params_dict)

        if env.is_universal:
            desc = UniversalActionSpace.describe(
                action,
                cont_params if cont_params is not None else UniversalActionSpace.neutral_params(),
                profile.resolution,
            )
        else:
            movement, action_type = env.action_space.decode(action)
            desc = f"{movement}_{action_type}"
        logger.info(f"Step {step}: {desc} (value={value:.3f})")

        # Auto-buy items (legacy HoK mode only — universal mode learns this via RL)
        if not env.is_universal and env.action_mapper is not None:
            if step > 0 and step % config.collection.auto_buy_interval == 0:
                for skill_name in ["加三技能", "加二技能", "加一技能", "购买"]:
                    cmd = env.action_mapper.get_action_command(skill_name)
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
