"""
Genshin Impact (原神) game profile.

Universal mode: the action space is phone touch primitives shared
across all games.  This profile only defines game understanding.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, ScreenRegion


class GenshinImpactProfile(GameProfile):
    """
    Profile for Genshin Impact (原神).

    Action mode: universal (7 touch types + 5 continuous params)
    State classes: 4 game phase types
    Resolution: 2560x1440 (landscape, tablet/high-end)
    """

    @property
    def display_name(self) -> str:
        return "原神 (Genshin Impact)"

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
        return 4

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            "defeat_enemy": 1.0,
            "defeat_boss": 5.0,
            "character_downed": -3.0,
            "party_wiped": -10.0,
            "chest_opened": 2.0,
            "material_collected": 0.3,
            "quest_completed": 10.0,
            "exploration": 0.005,
            "normal": 0.005,
            "other": -0.001,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["party_wiped"]

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
