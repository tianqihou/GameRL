"""
Honor of Kings (王者荣耀) game profile.

Defines the action space, state classes, touch mappings, and screen
regions specific to Honor of Kings. This is the default profile and
matches the original WZCQ project's configuration.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..utils.actions import Movement, ActionType
from .base import GameProfile, TouchAction, ScreenRegion


class HonorOfKingsProfile(GameProfile):
    """
    Profile for Honor of Kings (王者荣耀).

    Action space: 10 movements x 13 actions = 130 tokens
    BOS token: 128 (within vocab, corresponds to '无移动_无动作')
    State classes: 6 event types
    Resolution: 1080x2160 (typical phone)
    """

    @property
    def display_name(self) -> str:
        return "王者荣耀 (Honor of Kings)"

    @property
    def movements(self) -> List[str]:
        return [m.value for m in Movement]

    @property
    def actions(self) -> List[str]:
        return [a.value for a in ActionType]

    @property
    def bos_token(self) -> int:
        # '无移动_无动作' = Movement.NONE(9) * 13 + ActionType.NO_ACTION(12) = 129
        # But original project uses 128 as BOS. We keep 128 for compatibility.
        # Token 128 = Movement.STOP(8) * 13 + ActionType.NO_ACTION(12) = 116
        # Actually, let's compute: the original BOS_TOKEN=128 maps to
        # movement_idx=128//13=9 (NONE), action_idx=128%13=11 (SIGNAL_GATHER)
        # But the original code treats 128 as a special BOS, not as a regular action.
        # For backward compat, we keep BOS_TOKEN=128.
        return 128

    @property
    def idle_movements(self) -> List[str]:
        return [Movement.NONE.value, Movement.STOP.value]

    @property
    def idle_actions(self) -> List[str]:
        return [ActionType.NO_ACTION.value]

    @property
    def state_classes(self) -> List[str]:
        return [
            "kill_minion",
            "kill_tower",
            "kill_hero",
            "assist_kill",
            "attacked_by_tower",
            "killed",
            "death",
            "normal",
        ]

    @property
    def resolution(self) -> Tuple[int, int]:
        return (1080, 2160)

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            # Combat rewards
            "kill_minion": 2.0,
            "kill_tower": 5.0,
            "kill_hero": 5.0,
            "assist_kill": 2.0,
            # Penalties
            "attacked_by_tower": -0.5,
            "killed": -2.0,
            "death": -1.0,
            # Baseline
            "normal": 0.01,
            "other": -0.003,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["death"]

    @property
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Touch coordinates for 1080x2160 resolution."""
        joystick_center = (237, 321)
        joystick_radius = 112

        mapping: Dict[str, TouchAction] = {}

        # Movement directions (joystick moves from center to offset)
        movement_offsets = {
            Movement.UP.value: (0, joystick_radius),
            Movement.DOWN.value: (0, -60),
            Movement.LEFT.value: (-joystick_radius, 0),
            Movement.RIGHT.value: (0, joystick_radius),
            Movement.UP_LEFT.value: (78, -78),
            Movement.UP_RIGHT.value: (78, joystick_radius),
            Movement.DOWN_LEFT.value: (-78, -78),
            Movement.DOWN_RIGHT.value: (-78, joystick_radius),
        }

        for name, (dx, dy) in movement_offsets.items():
            cx, cy = joystick_center
            mapping[name] = TouchAction(
                type="joystick",
                coords=(cx, cy, cx + dx, cy + dy),
                duration_ms=100,
            )

        # Idle movements (just tap joystick center and release)
        for name in self.idle_movements:
            mapping[name] = TouchAction(
                type="tap",
                coords=joystick_center,
                duration_ms=300,
            )

        # Button actions
        button_coords = {
            ActionType.ATTACK.value: (169, 1982),
            ActionType.LAST_HIT.value: (106, 1822),
            ActionType.PUSH_TOWER.value: (318, 2067),
            ActionType.SKILL_1.value: (133, 1660),
            ActionType.SKILL_2.value: (342, 1782),
            ActionType.SKILL_3.value: (455, 1984),
            ActionType.SUMMONER.value: (117, 1496),
            ActionType.RECALL.value: (108, 1206),
            ActionType.HEAL.value: (111, 1345),
            ActionType.SIGNAL_ATTACK.value: (945, 2110),
            ActionType.SIGNAL_RETREAT.value: (851, 2112),
            ActionType.SIGNAL_GATHER.value: (765, 2110),
        }

        for name, coord in button_coords.items():
            mapping[name] = TouchAction(
                type="tap",
                coords=coord,
                duration_ms=100,
            )

        return mapping

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions for 1080x2160 resolution."""
        return {
            "hp_bar": ScreenRegion("hp_bar", x=20, y=20, width=200, height=30),
            "minimap": ScreenRegion("minimap", x=880, y=20, width=200, height=200),
            "skills": ScreenRegion("skills", x=100, y=1450, width=400, height=700),
            "gold": ScreenRegion("gold", x=850, y=180, width=120, height=30),
            "kill_feed": ScreenRegion("kill_feed", x=350, y=0, width=380, height=60),
        }
