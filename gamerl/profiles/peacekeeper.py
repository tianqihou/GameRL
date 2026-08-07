"""
Peacekeeper Elite (和平精英) game profile.

Example profile for a battle royale FPS game. Demonstrates how the
GameProfile abstraction adapts to a different genre with different
action space, state classes, and touch mappings.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, TouchAction, ScreenRegion


class PeacekeeperEliteProfile(GameProfile):
    """
    Profile for Peacekeeper Elite (和平精英).

    Action space: 8 movements x 10 actions = 80 tokens
    BOS token: 0 (first token, no special BOS needed for FPS)
    State classes: 5 game phase types
    Resolution: 2340x1080 (landscape)
    """

    @property
    def display_name(self) -> str:
        return "和平精英 (Peacekeeper Elite)"

    @property
    def movements(self) -> List[str]:
        return [
            "forward",
            "backward",
            "left",
            "right",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
        ]

    @property
    def actions(self) -> List[str]:
        return [
            "noop",
            "shoot",
            "aim",
            "reload",
            "crouch",
            "prone",
            "jump",
            "pickup",
            "switch_weapon",
            "use_item",
        ]

    @property
    def bos_token(self) -> int:
        return 0

    @property
    def idle_movements(self) -> List[str]:
        return []

    @property
    def idle_actions(self) -> List[str]:
        return ["noop"]

    @property
    def state_classes(self) -> List[str]:
        return [
            "parachuting",
            "looting",
            "combat",
            "driving",
            "final_circle",
        ]

    @property
    def resolution(self) -> Tuple[int, int]:
        return (2340, 1080)

    @property
    def continuous_params(self) -> List[str]:
        """FPS needs look/aim direction (dx, dy) in [-1, 1]."""
        return ["look_dx", "look_dy"]

    @property
    def detection_classes(self) -> List[str]:
        return ["player", "enemy", "vehicle", "loot_item", "weapon", "door", "safe_zone"]

    @property
    def num_skills(self) -> int:
        return 2  # primary weapon, secondary weapon

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            # Combat
            "kill_enemy": 10.0,
            "down_enemy": 3.0,         # downed but not killed
            "got_killed": -5.0,
            "teammate_died": -1.0,
            # Survival
            "survived_frame": 0.01,    # per-frame survival bonus
            "reached_final_circle": 5.0,
            "won_match": 20.0,
            # Looting
            "loot_item": 0.5,
            # Baseline
            "normal": 0.01,
            "other": -0.001,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["got_killed", "won_match"]

    @property
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Touch coordinates for 2340x1080 landscape resolution."""
        # Left joystick center for movement
        move_center = (400, 850)
        move_radius = 150

        mapping: Dict[str, TouchAction] = {}

        # Movement (left joystick)
        offsets = {
            "forward": (0, -move_radius),
            "backward": (0, move_radius),
            "left": (-move_radius, 0),
            "right": (move_radius, 0),
            "forward_left": (-move_radius // 2, -move_radius // 2),
            "forward_right": (move_radius // 2, -move_radius // 2),
            "backward_left": (-move_radius // 2, move_radius // 2),
            "backward_right": (move_radius // 2, move_radius // 2),
        }

        for name, (dx, dy) in offsets.items():
            cx, cy = move_center
            mapping[name] = TouchAction(
                type="joystick",
                coords=(cx, cy, cx + dx, cy + dy),
                duration_ms=100,
            )

        # Right side action buttons
        button_coords = {
            "shoot": (2050, 850),
            "reload": (1900, 500),
            "crouch": (600, 950),
            "prone": (600, 1000),
            "jump": (2100, 600),
            "pickup": (1700, 850),
            "switch_weapon": (1950, 400),
            "use_item": (1750, 550),
        }

        for name, coord in button_coords.items():
            mapping[name] = TouchAction(
                type="tap",
                coords=coord,
                duration_ms=100,
            )

        # Dynamic "aim" action: swipe from screen center towards (look_dx, look_dy)
        # The policy outputs look_dx, look_dy in [-1, 1]; the swipe end point
        # is computed at runtime as center + (dx, dy) * max_radius.
        screen_cx, screen_cy = 1170, 540  # center of 2340x1080
        mapping["aim"] = TouchAction(
            type="look",
            coords=(screen_cx, screen_cy, 400),  # center_x, center_y, max_radius
            duration_ms=50,
            param_keys=("look_dx", "look_dy"),
        )

        return mapping

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions for 2340x1080 landscape."""
        return {
            "hp_bar": ScreenRegion("hp_bar", x=50, y=50, width=200, height=20),
            "minimap": ScreenRegion("minimap", x=2050, y=50, width=250, height=250),
            "crosshair": ScreenRegion("crosshair", x=1020, y=440, width=300, height=200),
            "inventory": ScreenRegion("inventory", x=1500, y=100, width=400, height=300),
            "compass": ScreenRegion("compass", x=1000, y=20, width=340, height=40),
        }
