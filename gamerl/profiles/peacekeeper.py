"""
Peacekeeper Elite (和平精英) game profile.

Universal mode: the action space is phone touch primitives shared
across all games.  This profile only defines game understanding.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, ScreenRegion


class PeacekeeperEliteProfile(GameProfile):
    """
    Profile for Peacekeeper Elite (和平精英).

    Action mode: universal (7 touch types + 5 continuous params)
    State classes: 5 game phase types
    Resolution: 2340x1080 (landscape)
    """

    @property
    def display_name(self) -> str:
        return "和平精英 (Peacekeeper Elite)"

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
    def detection_classes(self) -> List[str]:
        return ["player", "enemy", "vehicle", "loot_item", "weapon", "door", "safe_zone"]

    @property
    def num_skills(self) -> int:
        return 2

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            "kill_enemy": 10.0,
            "down_enemy": 3.0,
            "got_killed": -5.0,
            "teammate_died": -1.0,
            "survived_frame": 0.01,
            "reached_final_circle": 5.0,
            "won_match": 20.0,
            "loot_item": 0.5,
            "normal": 0.01,
            "other": -0.001,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["got_killed", "won_match"]

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
