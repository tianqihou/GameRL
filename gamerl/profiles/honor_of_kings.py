"""
Honor of Kings (王者荣耀) game profile.

Universal mode: the action space is phone touch primitives shared
across all games.  This profile only defines game understanding
(what to detect, how to reward) — not how to interact.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, ScreenRegion


class HonorOfKingsProfile(GameProfile):
    """
    Profile for Honor of Kings (王者荣耀).

    Action mode: universal (7 touch types + 5 continuous params)
    State classes: 8 event types
    Resolution: 1080x2160 (typical phone, portrait)
    """

    @property
    def display_name(self) -> str:
        return "王者荣耀 (Honor of Kings)"

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
    def detection_classes(self) -> List[str]:
        return ["hero", "enemy", "minion", "tower", "hp_bar", "skill_button"]

    @property
    def num_skills(self) -> int:
        return 3

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            "kill_minion": 2.0,
            "kill_tower": 5.0,
            "kill_hero": 5.0,
            "assist_kill": 2.0,
            "attacked_by_tower": -0.5,
            "killed": -2.0,
            "death": -1.0,
            "normal": 0.01,
            "other": -0.003,
        }

    @property
    def terminal_events(self) -> List[str]:
        return ["death"]

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
