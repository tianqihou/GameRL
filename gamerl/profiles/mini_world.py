"""
Mini World (迷你世界) game profile.

Universal mode: the action space is phone touch primitives shared
across all games.  This profile only defines game understanding.

3D sandbox game similar to Minecraft. Features survival mode (resource
gathering, crafting, building, combat), creative mode (unlimited building),
and multiplayer.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, ScreenRegion


class MiniWorldProfile(GameProfile):
    """
    Profile for Mini World (迷你世界).

    Action mode: universal (7 touch types + 5 continuous params)
    State classes: 7 game phase types
    Resolution: 1920x1080 (landscape)
    """

    @property
    def display_name(self) -> str:
        return "迷你世界 (Mini World)"

    @property
    def state_classes(self) -> List[str]:
        return [
            "survival_day",
            "survival_night",
            "mining",
            "building",
            "combat",
            "inventory",
            "creative",
        ]

    @property
    def resolution(self) -> Tuple[int, int]:
        return (1920, 1080)

    @property
    def detection_classes(self) -> List[str]:
        return [
            "player",
            "mob",
            "resource_node",
            "drop_item",
            "block",
            "npc",
            "boss",
        ]

    @property
    def num_skills(self) -> int:
        return 3

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            "survival_day": 0.01,
            "survival_night": 0.02,
            "mining": 0.0,
            "building": 0.0,
            "combat": 0.0,
            "inventory": 0.0,
            "creative": 0.005,
            "collect_resource": 0.5,
            "harvest_crop": 0.3,
            "craft_item": 2.0,
            "upgrade_tool": 3.0,
            "place_block": 0.2,
            "defeat_mob": 2.0,
            "defeat_boss": 10.0,
            "take_damage": -0.5,
            "survive_night": 5.0,
            "starve": -1.0,
            "explore_new_area": 1.0,
            "tame_pet": 5.0,
            "normal": 0.01,
            "other": -0.002,
        }

    @property
    def terminal_events(self) -> List[str]:
        return []

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions for 1920x1080 landscape."""
        return {
            "hp_bar": ScreenRegion("hp_bar", x=40, y=40, width=200, height=25),
            "hunger_bar": ScreenRegion("hunger_bar", x=40, y=70, width=200, height=25),
            "hotbar": ScreenRegion("hotbar", x=600, y=950, width=600, height=80),
            "minimap": ScreenRegion("minimap", x=1700, y=40, width=180, height=180),
            "compass": ScreenRegion("compass", x=40, y=100, width=80, height=80),
        }
