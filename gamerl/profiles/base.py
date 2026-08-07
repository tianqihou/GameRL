"""
Abstract base class for game profiles.

A GameProfile encapsulates all game-specific configuration that the
generic RL infrastructure needs to interact with a particular game:

- Action space: What actions the agent can take (movements + buttons)
- State classes: What game states the classifier should detect
- Touch mapping: How abstract actions map to screen touch coordinates
- Screen regions: Named regions of interest (HP bar, minimap, etc.)
- Reward config: How different game events are rewarded
- Resolution: Default device resolution for this game

To support a new game, subclass GameProfile and implement the required
properties. The generic training/inference pipeline works with any
profile without modification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.actions import ActionSpace


@dataclass
class TouchAction:
    """A touch action definition.

    Types:
    - "tap": Fixed coordinate tap. coords = (x, y)
    - "joystick": Fixed-direction joystick slide. coords = (start_x, start_y, end_x, end_y)
    - "swipe": Fixed swipe gesture. coords = (x1, y1, x2, y2)
    - "look": Dynamic swipe whose direction is set at runtime by continuous
        parameters. coords = (center_x, center_y, max_radius).
        param_keys = [dx_key, dy_key]; values are in [-1, 1].
        End point = (center_x + dx * max_radius, center_y + dy * max_radius).
    - "dynamic_joystick": Same coordinate math as "look" but executed via
        the joystick touch protocol (pointer id 1).
    """

    type: str  # "tap" | "joystick" | "swipe" | "look" | "dynamic_joystick"
    coords: Tuple[int, ...]
    duration_ms: int = 100
    # Names of continuous params this action consumes (only for "look" / "dynamic_joystick").
    # Order matters: [dx_param, dy_param].
    param_keys: Tuple[str, ...] = ()


@dataclass
class ScreenRegion:
    """A named rectangular region of the screen."""

    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Return as (left, top, right, bottom)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class GameProfile(ABC):
    """
    Base class for game-specific profiles.

    Subclasses must define:
    - display_name: Human-readable game name
    - movements: List of movement action labels (strings)
    - actions: List of button/action labels (strings)
    - bos_token: Token ID for beginning-of-sequence (within vocab)
    - idle_movements: Movement labels that require no touch command
    - idle_actions: Action labels that require no touch command
    - state_classes: List of game state category names
    - touch_mapping: Dict mapping action labels to TouchAction objects
    - screen_regions: Dict of named ScreenRegion objects
    - resolution: Default screen resolution (width, height)
    - reward_events: Dict[str, float] mapping event names to reward weights
    - terminal_events: List of event names that signal episode end
    """

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable game name."""
        ...

    @property
    @abstractmethod
    def movements(self) -> List[str]:
        """Movement direction labels (e.g., ['上移', '下移', ...])."""
        ...

    @property
    @abstractmethod
    def actions(self) -> List[str]:
        """Button/action labels (e.g., ['攻击', '一技能', ...])."""
        ...

    @property
    @abstractmethod
    def bos_token(self) -> int:
        """Beginning-of-sequence token ID (must be within vocab range)."""
        ...

    @property
    @abstractmethod
    def idle_movements(self) -> List[str]:
        """Movement labels that require no touch command (e.g., ['无移动'])."""
        ...

    @property
    @abstractmethod
    def idle_actions(self) -> List[str]:
        """Action labels that require no touch command (e.g., ['无动作'])."""
        ...

    @property
    @abstractmethod
    def state_classes(self) -> List[str]:
        """Game state category names for the state classifier."""
        ...

    @property
    @abstractmethod
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Maps action/movement labels to touch coordinates."""
        ...

    @property
    @abstractmethod
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions of interest."""
        ...

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Default device screen resolution (width, height)."""
        ...

    @property
    def continuous_params(self) -> List[str]:
        """Names of continuous parameters the policy outputs for this game.

        Override in subclass to enable hybrid action space.  Each name
        corresponds to one scalar in [-1, 1] that the policy network
        produces alongside the discrete token.

        Examples:
        - Peacekeeper: ["look_dx", "look_dy"]  -- aim / look direction
        - Genshin:     ["aim_dx", "aim_dy"]     -- burst / bow aim
        - HoK:         []                        -- pure discrete (default)

        TouchActions of type "look" or "dynamic_joystick" reference these
        names via their ``param_keys`` field.
        """
        return []

    @property
    def num_continuous_params(self) -> int:
        """Number of continuous parameters (0 for pure-discrete games)."""
        return len(self.continuous_params)

    @property
    def is_hybrid(self) -> bool:
        """Whether this game uses a hybrid (discrete + continuous) action space."""
        return self.num_continuous_params > 0

    @property
    def detection_classes(self) -> List[str]:
        """
        Object detection class names for YOLO.

        Override in subclass to customize. Default classes are generic
        MOBA game objects.
        """
        return ["hero", "enemy", "minion", "tower", "hp_bar", "skill_button"]

    @property
    def num_skills(self) -> int:
        """Number of skill slots (for structured state encoding)."""
        return 3

    @property
    @abstractmethod
    def reward_events(self) -> Dict[str, float]:
        """Map of event name to reward weight for this game.

        Keys should include all state_classes (for per-frame event
        classification) plus any additional rewardable events.
        Every game must define at least 'normal' and 'other' keys.
        """
        ...

    @property
    def terminal_events(self) -> List[str]:
        """Event names that signal episode termination.

        Override in subclass to specify which events end the episode
        (e.g., 'death', 'match_won', 'party_wiped').
        """
        return []

    # ---- Derived properties (no need to override) ----

    @property
    def action_space(self) -> ActionSpace:
        """Construct an ActionSpace from this profile's movements and actions."""
        return ActionSpace(
            movements=self.movements,
            actions=self.actions,
            bos_token=self.bos_token,
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (movements * actions)."""
        return len(self.movements) * len(self.actions)

    @property
    def num_state_classes(self) -> int:
        """Number of state classification categories."""
        return len(self.state_classes)

    def is_idle_movement(self, movement: str) -> bool:
        """Check if a movement label is an idle/no-op movement."""
        return movement in self.idle_movements

    def is_idle_action(self, action: str) -> bool:
        """Check if an action label is an idle/no-op action."""
        return action in self.idle_actions

    def get_touch_action(self, label: str) -> Optional[TouchAction]:
        """Get the touch action for a movement or action label."""
        return self.touch_mapping.get(label)

    def to_dict(self) -> dict:
        """Serialize profile to a dictionary (for logging/debugging)."""
        return {
            "display_name": self.display_name,
            "movements": self.movements,
            "actions": self.actions,
            "vocab_size": self.vocab_size,
            "bos_token": self.bos_token,
            "state_classes": self.state_classes,
            "resolution": list(self.resolution),
            "idle_movements": self.idle_movements,
            "idle_actions": self.idle_actions,
            "reward_events": self.reward_events,
            "terminal_events": self.terminal_events,
            "continuous_params": self.continuous_params,
            "is_hybrid": self.is_hybrid,
        }
