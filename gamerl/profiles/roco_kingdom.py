"""
Roco Kingdom World (洛克王国世界) game profile.

Open-world pet collection RPG with turn-based combat. The agent explores
the world to discover and catch pets, battles NPCs and other players,
completes quests, and levels up their pet team. Combat is turn-based
with 4 skills per pet, an energy system, and type-effectiveness mechanics.

Action space: 5 movements x 10 actions = 50 tokens
BOS token: 0
State classes: 5 game phase types
Resolution: 1920x1080 (landscape)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, TouchAction, ScreenRegion


class RocoKingdomProfile(GameProfile):
    """
    Profile for Roco Kingdom World (洛克王国世界).

    Two-phase gameplay:
    - Exploration: move through the open world, encounter wild pets,
      interact with NPCs, collect resources, open chests.
    - Battle: turn-based combat with 4 skills, energy management,
      pet switching, catching, and fleeing.

    The action space is designed to cover both phases. During exploration,
    movement actions are active and skill buttons act as interactions.
    During battle, skill buttons are the primary actions.
    """

    @property
    def display_name(self) -> str:
        return "洛克王国世界 (Roco Kingdom)"

    @property
    def movements(self) -> List[str]:
        return [
            "up",
            "down",
            "left",
            "right",
            "stop",
        ]

    @property
    def actions(self) -> List[str]:
        return [
            "noop",
            "skill_1",
            "skill_2",
            "skill_3",
            "skill_4",
            "gather_energy",
            "catch_pet",
            "switch_pet",
            "flee",
            "interact",
        ]

    @property
    def bos_token(self) -> int:
        return 0

    @property
    def idle_movements(self) -> List[str]:
        return ["stop"]

    @property
    def idle_actions(self) -> List[str]:
        return ["noop"]

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
        return 4  # 4 skill buttons per pet

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            # Phase-based per-frame rewards (state classes)
            "exploration": 0.005,
            "battle": 0.0,
            "catching": 0.0,
            "menu": 0.0,
            "dialogue": 0.0,
            # Pet collection
            "catch_pet": 8.0,
            "discover_new_pet": 3.0,
            "pet_evolve": 10.0,
            "pet_levelup": 1.0,
            "learn_skill": 3.0,
            # Combat
            "defeat_enemy_pet": 2.0,
            "win_battle": 6.0,
            "defeat_world_boss": 15.0,
            "pet_downed": -2.0,
            "battle_lost": -8.0,
            # Exploration
            "open_chest": 3.0,
            "collect_resource": 0.5,
            "unlock_area": 5.0,
            # Quests
            "complete_quest": 8.0,
            "complete_daily": 2.0,
            # Baseline
            "normal": 0.01,
            "other": -0.002,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["battle_lost"]

    @property
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Touch coordinates for 1920x1080 landscape resolution."""
        move_center = (300, 800)
        move_radius = 140

        mapping: Dict[str, TouchAction] = {}

        # Movement (left joystick, used in exploration phase)
        offsets = {
            "up": (0, -move_radius),
            "down": (0, move_radius),
            "left": (-move_radius, 0),
            "right": (move_radius, 0),
        }

        for name, (dx, dy) in offsets.items():
            cx, cy = move_center
            mapping[name] = TouchAction(
                type="joystick",
                coords=(cx, cy, cx + dx, cy + dy),
                duration_ms=100,
            )

        # stop = release joystick (no touch needed, handled as idle)

        # Battle UI: 4 skill buttons on the left side
        skill_coords = {
            "skill_1": (200, 600),
            "skill_2": (200, 750),
            "skill_3": (350, 600),
            "skill_4": (350, 750),
        }

        for name, coord in skill_coords.items():
            mapping[name] = TouchAction(
                type="tap",
                coords=coord,
                duration_ms=100,
            )

        # Action buttons on the right side
        action_coords = {
            "gather_energy": (150, 900),     # bottom-left, energy recovery
            "catch_pet": (1750, 900),        # bottom-right
            "switch_pet": (1600, 900),       # next to catch
            "flee": (1850, 950),             # far bottom-right
            "interact": (1700, 600),         # mid-right, context action
        }

        for name, coord in action_coords.items():
            mapping[name] = TouchAction(
                type="tap",
                coords=coord,
                duration_ms=100,
            )

        return mapping

    @property
    def screen_regions(self) -> Dict[str, ScreenRegion]:
        """Named screen regions for 1920x1080 landscape."""
        return {
            # Battle UI
            "my_pet_hp": ScreenRegion("my_pet_hp", x=50, y=80, width=250, height=25),
            "enemy_pet_hp": ScreenRegion("enemy_pet_hp", x=1620, y=80, width=250, height=25),
            "energy_bar": ScreenRegion("energy_bar", x=50, y=110, width=200, height=20),
            "skill_panel": ScreenRegion("skill_panel", x=100, y=550, width=350, height=350),
            "battle_log": ScreenRegion("battle_log", x=1550, y=550, width=350, height=300),
            # Exploration UI
            "minimap": ScreenRegion("minimap", x=1700, y=40, width=180, height=180),
            "quest_tracker": ScreenRegion("quest_tracker", x=40, y=200, width=300, height=120),
            "pet_team": ScreenRegion("pet_team", x=1700, y=250, width=180, height=60),
        }
