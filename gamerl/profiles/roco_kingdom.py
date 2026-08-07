"""
Roco Kingdom World (洛克王国世界) game profile.

Universal mode: the action space is phone touch primitives shared
across all games.  This profile only defines game understanding.

Open-world pet collection RPG with turn-based combat. The agent explores
the world to discover and catch pets, battles NPCs and other players,
completes quests, and levels up their pet team.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, ScreenRegion


class RocoKingdomProfile(GameProfile):
    """
    Profile for Roco Kingdom World (洛克王国世界).

    Action mode: universal (7 touch types + 5 continuous params)
    State classes: 5 game phase types
    Resolution: 1920x1080 (landscape)
    """

    @property
    def display_name(self) -> str:
        return "洛克王国世界 (Roco Kingdom)"

    @property
    def state_classes(self) -> List[str]:
        return [
            "exploration",
            "battle",
            "catching",
            "menu",
            "dialogue",
        ]

    @property
    def resolution(self) -> Tuple[int, int]:
        return (1920, 1080)

    @property
    def detection_classes(self) -> List[str]:
        return [
            "player",
            "pet_wild",
            "pet_enemy",
            "npc",
            "chest",
            "resource",
            "boss",
        ]

    @property
    def num_skills(self) -> int:
        return 4

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            "exploration": 0.005,
            "battle": 0.0,
            "catching": 0.0,
            "menu": 0.0,
            "dialogue": 0.0,
            "catch_pet": 8.0,
            "discover_new_pet": 3.0,
            "pet_evolve": 10.0,
            "pet_levelup": 1.0,
            "learn_skill": 3.0,
            "defeat_enemy_pet": 2.0,
            "win_battle": 6.0,
            "defeat_world_boss": 15.0,
            "pet_downed": -2.0,
            "battle_lost": -8.0,
            "open_chest": 3.0,
            "collect_resource": 0.5,
            "unlock_area": 5.0,
            "complete_quest": 8.0,
            "complete_daily": 2.0,
            "normal": 0.01,
            "other": -0.002,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["battle_lost"]

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions for 1920x1080 landscape."""
        return {
            "my_pet_hp": ScreenRegion("my_pet_hp", x=50, y=80, width=250, height=25),
            "enemy_pet_hp": ScreenRegion("enemy_pet_hp", x=1620, y=80, width=250, height=25),
            "energy_bar": ScreenRegion("energy_bar", x=50, y=110, width=200, height=20),
            "skill_panel": ScreenRegion("skill_panel", x=100, y=550, width=350, height=350),
            "battle_log": ScreenRegion("battle_log", x=1550, y=550, width=350, height=300),
            "minimap": ScreenRegion("minimap", x=1700, y=40, width=180, height=180),
            "quest_tracker": ScreenRegion("quest_tracker", x=40, y=200, width=300, height=120),
            "pet_team": ScreenRegion("pet_team", x=1700, y=250, width=180, height=60),
        }
