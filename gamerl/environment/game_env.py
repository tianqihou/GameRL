"""
Game environment wrapper for Honor of Kings.

Combines screen capture, device control, and backbone feature extraction
into a unified game loop interface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..models.backbone import BackboneExtractor
from ..utils.actions import ActionSpace, BOS_TOKEN, UniversalActionSpace
from .capture import ScreenCapture
from .device import ADBDevice, ActionMapper, TouchExecutor
from .reward import RewardShaper

logger = logging.getLogger("gamerl.environment")


@dataclass
class GameState:
    """Represents the state of the game at a point in time."""

    image_features: np.ndarray  # (seq_len, feature_dim)
    action_history: np.ndarray  # (seq_len,)
    raw_image: Optional[Image.Image] = None


class GameEnvironment:
    """
    Game environment that manages the interaction loop.

    Handles:
    - Screenshot capture and feature extraction
    - Action execution on the Android device
    - State history management (sliding window)
    - Reward computation via RewardShaper (optional)

    **Action modes:**

    * **Universal** (default) — uses :class:`TouchExecutor` to execute
      phone touch primitives (tap, swipe, drag, …) at normalised screen
      coordinates.  ``step()`` receives ``(touch_type, continuous_params)``.
      The action history stores touch-type indices (0–6).

    * **Legacy** — uses :class:`ActionMapper` with per-game
      ``touch_mapping``.  ``step()`` receives a discrete token that is
      decoded into (movement, action) and looked up in the mapping.

    Args:
        capture: Screen capture backend.
        device: ADB device controller.
        backbone: Image feature extraction backbone.
        action_space: Action space definition (legacy mode).
        action_mapper: Action-to-touch-command mapper (legacy mode).
        profile: Optional game profile for idle action checks and mode detection.
        max_history: Maximum sequence length for state history.
        reward_shaper: Optional RewardShaper for computing rewards and done flags.
        touch_executor: Optional TouchExecutor (universal mode).  If not
            provided but profile is universal, one is created from the profile.
    """

    def __init__(
        self,
        capture: ScreenCapture,
        device: ADBDevice,
        backbone: BackboneExtractor,
        action_space: ActionSpace | None = None,
        action_mapper: ActionMapper | None = None,
        profile=None,
        max_history: int = 300,
        reward_shaper: Optional[RewardShaper] = None,
        touch_executor: TouchExecutor | None = None,
    ):
        self.capture = capture
        self.device = device
        self.backbone = backbone
        self.profile = profile
        self.max_history = max_history
        self.reward_shaper = reward_shaper

        # Detect action mode
        self.is_universal = profile is not None and profile.is_universal

        if self.is_universal:
            # Universal mode: use TouchExecutor
            self.touch_executor = touch_executor or TouchExecutor(
                device=device,
                resolution=profile.resolution if profile else (1080, 2160),
            )
            self.action_space = None
            self.action_mapper = None
            self._bos_token = UniversalActionSpace.BOS_TOKEN
        else:
            # Legacy mode: use ActionMapper
            self.action_space = action_space or (profile.action_space if profile else ActionSpace())
            self.action_mapper = action_mapper or (
                ActionMapper.from_profile(profile) if profile else ActionMapper({})
            )
            self.touch_executor = None
            self._bos_token = BOS_TOKEN

        # State history
        self._image_features: List[np.ndarray] = []
        self._action_history: List[int] = [self._bos_token]

    def reset(self) -> GameState:
        """Reset the environment and return initial state."""
        self._image_features.clear()
        self._action_history = [self._bos_token]

        # Capture initial frame
        img = self.capture.capture()
        features = self._extract_features(img)

        self._image_features.append(features)

        return self._get_state(raw_image=img)

    def step(
        self,
        action: int,
        continuous_params: Dict[str, float] | None = None,
    ) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute an action and return the new state.

        **Universal mode:** *action* is a :class:`TouchType` index (0–6)
        and *continuous_params* is a dict with keys ``x``, ``y``, ``dx``,
        ``dy``, ``duration`` (values in [-1, 1]).  The :class:`TouchExecutor`
        converts them to ADB touch commands.

        **Legacy mode:** *action* is a discrete token decoded into
        (movement, action_type) strings and looked up in the profile's
        ``touch_mapping``.  *continuous_params* carries runtime values
        for dynamic actions (``"look"``, ``"dynamic_joystick"``).

        Args:
            action: Discrete action token (touch-type index in universal mode).
            continuous_params: Optional dict of param name → value.

        Returns:
            Tuple of (state, reward, done, info).
        """
        if self.is_universal:
            self._step_universal(action, continuous_params)
        else:
            self._step_legacy(action, continuous_params)

        # Capture new frame
        img = self.capture.capture()
        features = self._extract_features(img)

        # Update history
        self._image_features.append(features)
        self._action_history.append(action)

        # Maintain sliding window
        if len(self._image_features) > self.max_history:
            self._image_features = self._image_features[-self.max_history:]
            self._action_history = self._action_history[-self.max_history:]

        state = self._get_state(raw_image=img)

        # Compute reward and done via RewardShaper
        if self.reward_shaper is not None:
            curr_features = self._image_features[-1]
            prev_features = self._image_features[-2] if len(self._image_features) >= 2 else None
            action_seq = np.array(self._action_history, dtype=np.int64)
            reward, done, event = self.reward_shaper.compute_reward(
                curr_features=curr_features,
                action=action,
                action_seq=action_seq,
                prev_features=prev_features,
            )
            info = {"event": event}
        else:
            reward = 0.0
            done = False
            info = {}

        return state, reward, done, info

    def _step_universal(
        self,
        touch_type: int,
        continuous_params: Dict[str, float] | None,
    ) -> None:
        """Execute a universal touch primitive on the device."""
        from ..utils.actions import UniversalActionSpace, TouchType

        # Convert dict params to ordered array
        if continuous_params is not None:
            params = np.array([
                continuous_params.get(name, 0.0)
                for name in UniversalActionSpace.CONTINUOUS_PARAMS
            ], dtype=np.float32)
        else:
            params = UniversalActionSpace.neutral_params()

        # Skip WAIT type — just sleep, no touch
        if touch_type != TouchType.WAIT.value:
            self.touch_executor.execute(touch_type, params)

    def _step_legacy(
        self,
        action: int,
        continuous_params: Dict[str, float] | None,
    ) -> None:
        """Execute a legacy per-game action on the device."""
        # Decode action
        movement, action_type = self.action_space.decode(action)

        # Execute movement (skip idle movements)
        should_move = True
        if self.profile is not None:
            should_move = not self.profile.is_idle_movement(movement)
        else:
            should_move = movement not in ("无移动", "移动停", "noop")

        if should_move:
            cmd = self.action_mapper.get_action_command(movement, continuous_params)
            if cmd:
                self.device.send_touch_command(cmd)

        # Execute action type (skip idle actions)
        should_act = True
        if self.profile is not None:
            should_act = not self.profile.is_idle_action(action_type)
        else:
            should_act = action_type not in ("无动作", "noop")

        if should_act:
            cmd = self.action_mapper.get_action_command(action_type, continuous_params)
            if cmd:
                self.device.send_touch_command(cmd)

    def _extract_features(self, img: Image.Image) -> np.ndarray:
        """Extract features from a PIL image using the backbone."""
        # Convert to tensor: (1, 3, H, W)
        arr = np.array(img)
        tensor = torch.from_numpy(arr).to(self.backbone.features.device if hasattr(self.backbone, 'features') else 'cpu')
        tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        features = self.backbone(tensor)  # (1, grid*grid, C)
        return features.squeeze(0).cpu().numpy()  # (grid*grid, C)

    def _get_state(self, raw_image: Optional[Image.Image] = None) -> GameState:
        """Construct current game state from history."""
        # Pad sequences to same length
        seq_len = len(self._image_features)
        image_features = np.stack(self._image_features)  # (seq_len, grid*grid, C)
        action_history = np.array(self._action_history, dtype=np.int64)

        # Flatten image features for compatibility
        feature_dim = image_features.shape[-1] * image_features.shape[-2]
        image_features_flat = image_features.reshape(seq_len, -1)

        return GameState(
            image_features=image_features_flat,
            action_history=action_history,
            raw_image=raw_image,
        )

    def close(self) -> None:
        """Clean up resources."""
        self.capture.close()
