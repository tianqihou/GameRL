"""
Action space definition for game AI.

The action space is a combination of movement direction and action type,
encoded as a single discrete token. This mirrors the original project's
"方向_动作" vocabulary but with a clean, documented API.

The ActionSpace class is game-agnostic and works with string labels.
Game-specific enums (Movement, ActionType) are provided for backward
compatibility with the Honor of Kings profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class Movement(Enum):
    """Movement directions (joystick) - Honor of Kings specific."""

    UP = "上移"
    DOWN = "下移"
    LEFT = "左移"
    RIGHT = "右移"
    UP_LEFT = "左上移"
    UP_RIGHT = "右上移"
    DOWN_LEFT = "左下移"
    DOWN_RIGHT = "右下移"
    STOP = "移动停"
    NONE = "无移动"


class ActionType(Enum):
    """Action types (buttons) - Honor of Kings specific."""

    ATTACK = "攻击"
    LAST_HIT = "补刀"
    PUSH_TOWER = "推塔"
    SKILL_1 = "一技能"
    SKILL_2 = "二技能"
    SKILL_3 = "三技能"
    SUMMONER = "召唤师技能"
    RECALL = "回城"
    SIGNAL_ATTACK = "发起进攻"
    SIGNAL_RETREAT = "发起撤退"
    SIGNAL_GATHER = "发起集合"
    NO_ACTION = "无动作"
    HEAL = "恢复"


# Special tokens (backward compatibility)
PAD_TOKEN = -1
BOS_TOKEN = 128  # Honor of Kings BOS - within the 130-token vocab
NUM_SPECIAL_TOKENS = 0


@dataclass
class ActionSpace:
    """
    Discrete action space combining movement and action type.

    Total actions = len(movements) * len(actions)
    Works with any game's string-based action labels.

    Args:
        movements: List of movement direction labels (strings).
        actions: List of action/button labels (strings).
        bos_token: Token ID for beginning-of-sequence (within vocab range).
    """

    movements: List[str] = field(default_factory=lambda: [m.value for m in Movement])
    actions: List[str] = field(default_factory=lambda: [a.value for a in ActionType])
    bos_token: int = BOS_TOKEN

    @property
    def size(self) -> int:
        """Total number of action tokens (excluding special)."""
        return len(self.movements) * len(self.actions)

    @property
    def vocab_size(self) -> int:
        """Full vocabulary size including special tokens."""
        return self.size + NUM_SPECIAL_TOKENS

    def encode(self, movement: str, action: str) -> int:
        """Encode a (movement, action) pair into a single token."""
        m_idx = self.movements.index(movement)
        a_idx = self.actions.index(action)
        return m_idx * len(self.actions) + a_idx

    def decode(self, token: int) -> Tuple[str, str]:
        """Decode a token back into (movement, action) as strings."""
        m_idx = token // len(self.actions)
        a_idx = token % len(self.actions)
        return self.movements[m_idx], self.actions[a_idx]

    def decode_to_str(self, token: int) -> str:
        """Decode a token to the original '方向_动作' string format."""
        if token == self.bos_token:
            m, a = self.decode(token)
            return f"{m}_{a}"
        m, a = self.decode(token)
        return f"{m}_{a}"

    def build_vocab(self) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Build word-to-index and index-to-word mappings.

        Returns:
            Tuple of (word_to_idx, idx_to_word) dictionaries.
        """
        word_to_idx: Dict[str, int] = {}
        idx_to_word: Dict[str, str] = {}
        for m in self.movements:
            for a in self.actions:
                key = f"{m}_{a}"
                idx = self.encode(m, a)
                word_to_idx[key] = idx
                idx_to_word[str(idx)] = key
        return word_to_idx, idx_to_word

    def save_vocab(self, path: str | Path) -> None:
        """Save vocabulary to JSON files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        word_to_idx, idx_to_word = self.build_vocab()
        with open(path / "action_vocab.json", "w", encoding="utf-8") as f:
            json.dump(word_to_idx, f, ensure_ascii=False, indent=2)
        with open(path / "action_idx.json", "w", encoding="utf-8") as f:
            json.dump(idx_to_word, f, ensure_ascii=False, indent=2)


def default_action_space() -> ActionSpace:
    """Create the default action space for Honor of Kings."""
    return ActionSpace()


class HybridActionSpace:
    """
    Hybrid action space: discrete token + continuous parameters.

    Wraps a discrete :class:`ActionSpace` and augments it with an optional
    set of continuous parameters (each a scalar in ``[-1, 1]``).  This lets
    the policy control both *what* button to press (discrete) and *how* to
    perform dynamic actions like aiming or camera rotation (continuous).

    Games with ``continuous_params = []`` behave identically to a pure
    :class:`ActionSpace` — the hybrid wrapper is a transparent no-op.

    Args:
        discrete: The underlying discrete action space.
        continuous_param_names: Ordered list of continuous parameter names
            (e.g. ``["look_dx", "look_dy"]``).  May be empty.
    """

    def __init__(
        self,
        discrete: ActionSpace,
        continuous_param_names: List[str] | None = None,
    ):
        self.discrete = discrete
        self.continuous_param_names: List[str] = continuous_param_names or []
        self.continuous_dim = len(self.continuous_param_names)

    # ---- delegate to discrete ----

    @property
    def vocab_size(self) -> int:
        return self.discrete.vocab_size

    @property
    def size(self) -> int:
        return self.discrete.size

    def encode(self, movement: str, action: str) -> int:
        return self.discrete.encode(movement, action)

    def decode(self, token: int) -> Tuple[str, str]:
        return self.discrete.decode(token)

    # ---- continuous helpers ----

    def is_hybrid(self) -> bool:
        """True when continuous parameters are present."""
        return self.continuous_dim > 0

    def clamp_continuous(self, params: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Clamp continuous parameters to [-1, 1].

        Works with both numpy arrays and torch tensors (preserving type).
        """
        if isinstance(params, torch.Tensor):
            return params.clamp(-1.0, 1.0)
        return np.clip(params, -1.0, 1.0)

    def sample_continuous(self) -> np.ndarray:
        """Sample continuous params uniformly from [-1, 1].

        Useful for random baselines and sanity checks.
        """
        if self.continuous_dim == 0:
            return np.zeros(0, dtype=np.float32)
        return np.random.uniform(-1.0, 1.0, size=self.continuous_dim).astype(np.float32)

    def params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """Convert a continuous param vector to a name→value dict."""
        return {
            name: float(params[i])
            for i, name in enumerate(self.continuous_param_names)
        }

    def dict_to_params(self, d: Dict[str, float]) -> np.ndarray:
        """Convert a name→value dict back to a param vector."""
        return np.array(
            [d.get(name, 0.0) for name in self.continuous_param_names],
            dtype=np.float32,
        )

    @classmethod
    def from_profile(cls, profile) -> "HybridActionSpace":
        """Build a HybridActionSpace from a GameProfile."""
        return cls(
            discrete=profile.action_space,
            continuous_param_names=profile.continuous_params,
        )


# ---------------------------------------------------------------------------
# Universal Action Space — phone touch primitives (game-agnostic)
# ---------------------------------------------------------------------------

class TouchType(Enum):
    """Phone touch primitive types — universal across ALL games.

    These are the fundamental operations a touchscreen supports.
    The policy network always outputs one of these 7 types plus
    5 continuous parameters (x, y, dx, dy, duration) that specify
    *where* and *how* to perform the touch.
    """

    TAP = 0           # Quick touch-and-release at (x, y)
    LONG_PRESS = 1    # Hold at (x, y) for `duration` ms
    SWIPE = 2         # Touch at (x, y), slide to (x+dx, y+dy), release
    DRAG = 3          # Touch at (x, y), slide to (x+dx, y+dy), hold (joystick-style)
    DOUBLE_TAP = 4    # Two rapid taps at (x, y)
    KEY_EVENT = 5     # Hardware key (back/home/menu), selected by `x` param
    WAIT = 6          # No touch, wait for `duration` ms


# Hardware keys mappable via the KEY_EVENT touch type.
# The continuous `x` param in [-1, 1] is mapped to an index into this list.
HARDWARE_KEYS = [
    "KEYCODE_BACK",
    "KEYCODE_HOME",
    "KEYCODE_APP_SWITCH",
    "KEYCODE_VOLUME_UP",
    "KEYCODE_VOLUME_DOWN",
]


class UniversalActionSpace:
    """Universal action space based on phone touch primitives.

    This is the **same for every game**.  The policy network always
    outputs:

    * **7 discrete logits** — one per :class:`TouchType`
    * **5 continuous parameters** (all in ``[-1, 1]``):
      ``x``, ``y``, ``dx``, ``dy``, ``duration``

    At execution time the :class:`~gamerl.environment.device.TouchExecutor`
    converts these normalized values into real screen coordinates and
    ADB commands using the device resolution.

    Parameter mapping (all inputs in [-1, 1]):
      x        -> pixel_x = (x + 1) / 2 * screen_width
      y        -> pixel_y = (y + 1) / 2 * screen_height
      dx       -> pixel_dx = dx * min(w, h) / 2
      dy       -> pixel_dy = dy * min(w, h) / 2
      duration -> ms = (duration + 1) / 2 * 1950 + 50   (50ms to 2000ms)

    This completely replaces the per-game ``ActionSpace`` +
    ``touch_mapping`` approach.  Games no longer need to define
    movements, actions, or button coordinates — the policy learns
    *where* to touch through reinforcement learning.
    """

    DISCRETE_SIZE: int = len(TouchType)          # 7
    CONTINUOUS_PARAMS: List[str] = ["x", "y", "dx", "dy", "duration"]
    CONTINUOUS_DIM: int = 5
    BOS_TOKEN: int = TouchType.WAIT.value        # neutral starting action

    # Aliases (plain class attributes, not @classmethod @property)
    vocab_size: int = DISCRETE_SIZE
    size: int = DISCRETE_SIZE

    # duration range in milliseconds
    MIN_DURATION_MS: int = 50
    MAX_DURATION_MS: int = 2000

    @staticmethod
    def decode_params(
        params,
        resolution: Tuple[int, int],
    ) -> Tuple[int, int, int, int, int]:
        """Convert normalized [-1, 1] params to pixel coordinates.

        Args:
            params: Array-like of 5 values in [-1, 1].
            resolution: (width, height) of the device screen.

        Returns:
            Tuple of (pixel_x, pixel_y, pixel_dx, pixel_dy, duration_ms).
        """
        w, h = resolution
        half_min = min(w, h) / 2

        px = int((float(params[0]) + 1.0) / 2.0 * w)
        py = int((float(params[1]) + 1.0) / 2.0 * h)
        pdx = int(float(params[2]) * half_min)
        pdy = int(float(params[3]) * half_min)
        dur = int((float(params[4]) + 1.0) / 2.0 *
                  (UniversalActionSpace.MAX_DURATION_MS -
                   UniversalActionSpace.MIN_DURATION_MS) +
                  UniversalActionSpace.MIN_DURATION_MS)

        # Clamp to screen bounds
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))

        return px, py, pdx, pdy, dur

    @staticmethod
    def decode_key_index(x_param: float) -> int:
        """Map the x continuous param to a hardware key index."""
        idx = int((x_param + 1.0) / 2.0 * len(HARDWARE_KEYS))
        return max(0, min(len(HARDWARE_KEYS) - 1, idx))

    @staticmethod
    def sample() -> Tuple[int, np.ndarray]:
        """Sample a random universal action.

        Returns:
            Tuple of (touch_type_idx, continuous_params) where
            continuous_params is a float32 array of shape (5,).
        """
        touch_type = np.random.randint(0, UniversalActionSpace.DISCRETE_SIZE)
        params = np.random.uniform(-1.0, 1.0, size=UniversalActionSpace.CONTINUOUS_DIM).astype(np.float32)
        return int(touch_type), params

    @staticmethod
    def clamp_continuous(params) -> np.ndarray:
        """Clamp continuous parameters to [-1, 1]."""
        if torch is not None and isinstance(params, torch.Tensor):
            return params.clamp(-1.0, 1.0)
        return np.clip(params, -1.0, 1.0)

    @staticmethod
    def neutral_params() -> np.ndarray:
        """Return neutral continuous params (screen center, no movement, short duration)."""
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    @staticmethod
    def describe(touch_type: int, params, resolution: Tuple[int, int]) -> str:
        """Human-readable description of a universal action."""
        px, py, pdx, pdy, dur = UniversalActionSpace.decode_params(params, resolution)
        name = TouchType(touch_type).name
        return (
            f"{name}(x={px}, y={py}, dx={pdx}, dy={pdy}, dur={dur}ms)"
        )
