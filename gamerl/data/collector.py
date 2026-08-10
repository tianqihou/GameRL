"""
Training data collector.

Collects gameplay data (screenshots + actions) for offline training.
Supports both AI-driven and human-assisted data collection.

Replaces the 训练数据截取_A.py and 取训练数据.py from the original project.

**Universal mode:** records touch_type (0-6) + 5 continuous params per step.
**Legacy mode:** records movement/action labels + discrete action token.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..agent.ppo import PPOAgent
from ..environment.game_env import GameEnvironment
from ..utils.actions import BOS_TOKEN, TouchType, UniversalActionSpace

logger = logging.getLogger("gamerl.data")


class DataCollector:
    """
    Collect training data during gameplay.

    Records screenshots and actions to disk for later preprocessing.
    Supports manual override (human takes control) which is recorded
    alongside AI actions for semi-supervised learning.

    The action mode (universal / legacy) is auto-detected from the
    environment's profile.

    Args:
        env: Game environment.
        agent: PPO agent (for AI-driven collection).
        save_dir: Directory to save collected data.
    """

    def __init__(
        self,
        env: GameEnvironment,
        agent: Optional[PPOAgent],
        save_dir: str | Path,
    ):
        self.env = env
        self.agent = agent
        self.save_dir = Path(save_dir)

    @property
    def _is_universal(self) -> bool:
        return self.env.is_universal

    @property
    def _bos_token(self) -> int:
        return UniversalActionSpace.BOS_TOKEN if self._is_universal else BOS_TOKEN

    def _manual_default_action(self) -> tuple[int, np.ndarray | None]:
        """Default no-op action for manual mode."""
        if self._is_universal:
            return TouchType.WAIT.value, UniversalActionSpace.neutral_params()
        # Legacy: (NONE movement, NO_ACTION)
        action_space = self.env.action_space
        return action_space.encode("无移动", "无动作"), None

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
            auto_buy_interval: Automatically buy items every N steps (legacy mode only).
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

        logger.info(f"Starting data collection in {episode_dir}")

        for step in range(max_steps):
            loop_start = time.time()

            # Select action
            log_prob = 0.0
            value = 0.0
            cont_params: np.ndarray | None = None

            if self.agent and not manual_override:
                action, log_prob, value, cont_params, _ = self.agent.select_action(
                    state.image_features,
                    state.action_history,
                )
            else:
                # In manual mode, default to no-op
                # (external tooling can inject real human actions)
                action, cont_params = self._manual_default_action()

            # Convert continuous params to dict for env.step
            params_dict = None
            if cont_params is not None:
                params_dict = {
                    name: float(cont_params[i])
                    for i, name in enumerate(UniversalActionSpace.CONTINUOUS_PARAMS)
                }

            # Execute action
            state, reward, done, info = self.env.step(action, params_dict)

            # Save screenshot
            if state.raw_image is not None:
                img_path = episode_dir / f"{step}.jpg"
                state.raw_image.save(img_path)

            # Record action data (mode-specific format)
            record = {
                "frame": step,
                "image": f"{step}.jpg",
                "action_token": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "done": done,
                "timestamp": time.time(),
            }
            if self._is_universal:
                record["touch_type"] = action
                record["continuous_params"] = (
                    cont_params.tolist()
                    if cont_params is not None
                    else UniversalActionSpace.neutral_params().tolist()
                )
            else:
                movement, action_type = self.env.action_space.decode(action)
                record["movement"] = movement
                record["action"] = action_type

            actions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            actions_file.flush()

            # Auto-buy items periodically (legacy HoK mode only)
            if not self._is_universal and step > 0 and step % auto_buy_interval == 0:
                self._auto_buy()

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
        """Automatically buy items and upgrade skills (legacy HoK mode only)."""
        mapper = getattr(self.env, "action_mapper", None)
        if mapper is None:
            return
        # Buy item
        cmd = mapper.get_action_command("购买")
        if cmd:
            self.env.device.send_touch_command(cmd)
        # Upgrade skills
        for skill in ["加三技能", "加二技能", "加一技能"]:
            cmd = mapper.get_action_command(skill)
            if cmd:
                self.env.device.send_touch_command(cmd)
