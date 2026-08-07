"""
Training data collector.

Collects gameplay data (screenshots + actions) for offline training.
Supports both AI-driven and human-assisted data collection.

Replaces the 训练数据截取_A.py and 取训练数据.py from the original project.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..agent.ppo import PPOAgent
from ..environment.game_env import GameEnvironment
from ..utils.actions import ActionSpace, BOS_TOKEN

logger = logging.getLogger("gamerl.data")


class DataCollector:
    """
    Collect training data during gameplay.

    Records screenshots and actions to disk for later preprocessing.
    Supports manual override (human takes control) which is recorded
    alongside AI actions for semi-supervised learning.

    Args:
        env: Game environment.
        agent: PPO agent (for AI-driven collection).
        action_space: Action space.
        save_dir: Directory to save collected data.
    """

    def __init__(
        self,
        env: GameEnvironment,
        agent: Optional[PPOAgent],
        action_space: ActionSpace,
        save_dir: str | Path,
    ):
        self.env = env
        self.agent = agent
        self.action_space = action_space
        self.save_dir = Path(save_dir)

    def collect_episode(
        self,
        max_steps: int = 10000,
        auto_buy_interval: int = 50,
        target_fps: int = 5,
        manual_override: bool = False,
    ) -> str:
        """
        Collect a single episode of gameplay data.

        Args:
            max_steps: Maximum number of steps before stopping.
            auto_buy_interval: Automatically buy items every N steps.
            target_fps: Target frames per second for game interaction.
            manual_override: If True, wait for human input instead of using AI.

        Returns:
            Path to the collected data directory.
        """
        # Create episode directory
        episode_dir = self.save_dir / str(int(time.time()))
        episode_dir.mkdir(parents=True, exist_ok=True)

        actions_file = open(episode_dir / "actions.jsonl", "w", encoding="utf-8")
        frame_interval = 1.0 / target_fps

        state = self.env.reset()
        step = 0

        # BOS action
        last_action = BOS_TOKEN

        logger.info(f"Starting data collection in {episode_dir}")

        for step in range(max_steps):
            loop_start = time.time()

            # Select action
            if self.agent and not manual_override:
                action, log_prob, value = self.agent.select_action(
                    state.image_features,
                    state.action_history,
                )
            else:
                # In manual mode, action comes from keyboard input
                # (handled externally, default to no-op)
                action = self.action_space.encode(
                    self.action_space.movements[9],  # NONE
                    self.action_space.actions[11],    # NO_ACTION
                )

            # Execute action
            state, reward, done, info = self.env.step(action)

            # Save screenshot
            if state.raw_image is not None:
                img_path = episode_dir / f"{step}.jpg"
                state.raw_image.save(img_path)

            # Record action data
            movement, action_type = self.action_space.decode(action)
            record = {
                "frame": step,
                "image": f"{step}.jpg",
                "movement": movement.value,
                "action": action_type.value,
                "action_token": action,
                "log_prob": log_prob if self.agent else 0.0,
                "value": value if self.agent else 0.0,
                "reward": reward,
                "done": done,
                "timestamp": time.time(),
            }
            actions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            actions_file.flush()

            # Auto-buy items periodically
            if step > 0 and step % auto_buy_interval == 0:
                self._auto_buy()

            last_action = action

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if done:
                break

            if step % 100 == 0:
                logger.info(f"Step {step}/{max_steps}, FPS: {1.0 / max(elapsed, 0.001):.1f}")

        actions_file.close()
        logger.info(f"Collected {step + 1} frames in {episode_dir}")

        return str(episode_dir)

    def _auto_buy(self) -> None:
        """Automatically buy items and upgrade skills."""
        from ..environment.device import ActionMapper
        if hasattr(self.env, "action_mapper"):
            mapper = self.env.action_mapper
            # Buy item
            cmd = mapper.get_action_command("购买")
            if cmd:
                self.env.device.send_touch_command(cmd)
            # Upgrade skills
            for skill in ["加三技能", "加二技能", "加一技能"]:
                cmd = mapper.get_action_command(skill)
                if cmd:
                    self.env.device.send_touch_command(cmd)
