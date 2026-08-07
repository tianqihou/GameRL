"""
Genshin Impact (原神) game profile.

Example profile for an open-world action RPG. Demonstrates adaptation
to a game with exploration, combat, and menu navigation states.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, TouchAction, ScreenRegion


class GenshinImpactProfile(GameProfile):
    """
    Profile for Genshin Impact (原神).

    Action space: 8 movements x 9 actions = 72 tokens
    BOS token: 0
    State classes: 4 game phase types
    Resolution: 2560x1440 (landscape, tablet/high-end)
    """

    @property
    def display_name(self) -> str:
        return "原神 (Genshin Impact)"

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
            "normal_attack",
            "charged_attack",
            "elemental_skill",
            "elemental_burst",
            "sprint",
            "jump",
            "switch_char",
            "interact",
            "aim",  # dynamic look action (uses aim_dx, aim_dy)
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
    def continuous_params(self) -> List[str]:
        """Genshin needs aim direction for bow aiming and burst direction."""
        return ["aim_dx", "aim_dy"]

    @property
    def state_classes(self) -> List[str]:
        return [
            "overworld",
            "combat",
            "menu",
            "dialogue",
        ]

    @property
    def resolution(self) -> Tuple[int, int]:
        return (2560, 1440)

    @property
    def detection_classes(self) -> List[str]:
        return ["player", "enemy", "npc", "chest", "resource", "boss"]

    @property
    def num_skills(self) -> int:
        return 4  # normal attack, elemental skill, elemental burst, charged attack

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            # Combat
            "defeat_enemy": 1.0,
            "defeat_boss": 5.0,
            "character_downed": -3.0,
            "party_wiped": -10.0,
            # Exploration
            "chest_opened": 2.0,
            "material_collected": 0.3,
            "quest_completed": 10.0,
            "exploration": 0.005,      # new area discovered
            # Baseline
            "normal": 0.005,
            "other": -0.001,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["party_wiped"]

    @property
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Touch coordinates for 2560x1440 landscape resolution."""
        move_center = (450, 1100)
        move_radius = 180

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
            "normal_attack": (2200, 1150),
            "charged_attack": (2200, 1150),  # long press same button
            "elemental_skill": (2050, 1000),
            "elemental_burst": (2350, 950),
            "sprint": (700, 1250),
            "jump": (2300, 1050),
            "switch_char": (1200, 100),
            "interact": (1900, 1150),
        }

        for name, coord in button_coords.items():
            touch_type = "tap"
            duration = 100
            if name == "charged_attack":
                duration = 500  # long press for charged attack
            mapping[name] = TouchAction(
                type=touch_type,
                coords=coord,
                duration_ms=duration,
            )

        # Dynamic "aim" action: swipe from screen center towards (aim_dx, aim_dy)
        # Used for bow aiming and burst direction control.
        screen_cx, screen_cy = 1280, 720  # center of 2560x1440
        mapping["aim"] = TouchAction(
            type="look",
            coords=(screen_cx, screen_cy, 500),  # center_x, center_y, max_radius
            duration_ms=50,
            param_keys=("aim_dx", "aim_dy"),
        )

        return mapping

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions for 2560x1440 landscape."""
        return {
            "hp_bar": ScreenRegion("hp_bar", x=60, y=60, width=300, height=30),
            "energy": ScreenRegion("energy", x=60, y=100, width=200, height=20),
            "party": ScreenRegion("party", x=1000, y=40, width=560, height=80),
            "minimap": ScreenRegion("minimap", x=2280, y=60, width=220, height=220),
            "skill_icons": ScreenRegion("skill_icons", x=1900, y=900, width=500, height=300),
        }
