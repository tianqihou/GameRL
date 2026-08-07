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

from ..utils.actions import ActionSpace, UniversalActionSpace


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

    **Action modes:**

    * ``"universal"`` (default) — the action space is the fixed set of
      phone touch primitives (:class:`UniversalActionSpace`).  The policy
      always outputs 7 discrete touch types + 5 continuous params.
      The profile does **not** need to define movements, actions, or
      touch_mapping.  This is the recommended mode for new games.

    * ``"legacy"`` — the profile defines per-game movements, actions,
      and touch_mapping (the original per-game action space).  Use this
      only for backward compatibility with existing trained models.

    Subclasses must define (both modes):
    - display_name: Human-readable game name
    - state_classes: List of game state category names
    - reward_events: Dict[str, float] mapping event names to reward weights
    - resolution: Default screen resolution (width, height)

    Subclasses must additionally define (legacy mode only):
    - movements: List of movement action labels
    - actions: List of button/action labels
    - bos_token: Token ID for beginning-of-sequence
    - idle_movements: Movement labels that require no touch command
    - idle_actions: Action labels that require no touch command
    - touch_mapping: Dict mapping action labels to TouchAction objects
    - screen_regions: Dict of named ScreenRegion objects
    """

    @property
    def action_mode(self) -> str:
        """Action space mode: 'universal' (default) or 'legacy'.

        Override to 'legacy' in subclass to use per-game action space.
        """
        return "universal"

    @property
    def is_universal(self) -> bool:
        """True when this profile uses the universal action space."""
        return self.action_mode == "universal"

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable game name."""
        ...

    # ---- Legacy action properties (only needed when action_mode == 'legacy') ----

    @property
    def movements(self) -> List[str]:
        """Movement direction labels (legacy mode only)."""
        if self.is_universal:
            return []
        raise NotImplementedError("Override 'movements' for legacy action mode")

    @property
    def actions(self) -> List[str]:
        """Button/action labels (legacy mode only)."""
        if self.is_universal:
            return []
        raise NotImplementedError("Override 'actions' for legacy action mode")

    @property
    def bos_token(self) -> int:
        """Beginning-of-sequence token ID."""
        if self.is_universal:
            return UniversalActionSpace.BOS_TOKEN
        raise NotImplementedError("Override 'bos_token' for legacy action mode")

    @property
    def idle_movements(self) -> List[str]:
        """Movement labels that require no touch command (legacy mode only)."""
        return []

    @property
    def idle_actions(self) -> List[str]:
        """Action labels that require no touch command (legacy mode only)."""
        return []

    @property
    @abstractmethod
    def state_classes(self) -> List[str]:
        """Game state category names for the state classifier."""
        ...

    @property
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Maps action/movement labels to touch coordinates (legacy mode only)."""
        return {}

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions of interest. Override for game-specific regions."""
        return {}

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Default device screen resolution (width, height)."""
        ...

    @property
    def continuous_params(self) -> List[str]:
        """Names of continuous parameters the policy outputs.

        In **universal** mode this is always the 5 phone-touch params
        (x, y, dx, dy, duration).  In **legacy** mode, override to
        specify game-specific continuous params (e.g. aim direction).
        """
        if self.is_universal:
            return list(UniversalActionSpace.CONTINUOUS_PARAMS)
        return []

    @property
    def num_continuous_params(self) -> int:
        """Number of continuous parameters."""
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
        """Construct an ActionSpace from this profile's movements and actions.

        Only meaningful in legacy mode.  In universal mode, use
        :attr:`universal_action_space` instead.
        """
        return ActionSpace(
            movements=self.movements,
            actions=self.actions,
            bos_token=self.bos_token,
        )

    @property
    def universal_action_space(self) -> "UniversalActionSpace":
        """The universal action space (same for all games)."""
        return UniversalActionSpace()

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size.

        In universal mode this is always 7 (touch types).
        In legacy mode it's len(movements) * len(actions).
        """
        if self.is_universal:
            return UniversalActionSpace.DISCRETE_SIZE
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
            "action_mode": self.action_mode,
            "vocab_size": self.vocab_size,
            "bos_token": self.bos_token,
            "state_classes": self.state_classes,
            "resolution": list(self.resolution),
            "continuous_params": self.continuous_params,
            "is_hybrid": self.is_hybrid,
            "reward_events": self.reward_events,
            "terminal_events": self.terminal_events,
        }
