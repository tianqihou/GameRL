"""
Mini World (迷你世界) game profile.

3D sandbox game similar to Minecraft. Features survival mode (resource
gathering, crafting, building, combat), creative mode (unlimited building),
and multiplayer. Real-time action controls with joystick movement and
action buttons for break/place/jump.

Action space: 8 movements x 8 actions = 64 tokens
BOS token: 0
State classes: 7 game phase types
Resolution: 1920x1080 (landscape)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import GameProfile, TouchAction, ScreenRegion


class MiniWorldProfile(GameProfile):
    """
    Profile for Mini World (迷你世界).

    Supports both survival and creative modes. The RL agent learns to
    gather resources, craft tools, build structures, and survive
    encounters with hostile mobs at night.
    """

    @property
    def display_name(self) -> str:
        return "迷你世界 (Mini World)"

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
            "jump",
            "break_block",
            "place_block",
            "use_food",
            "switch_slot_1",
            "switch_slot_2",
            "open_inventory",
            "look",  # dynamic camera rotation (uses look_dx, look_dy)
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
        """Sandbox game needs camera look direction for building and exploring."""
        return ["look_dx", "look_dy"]

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
        return 3  # break, place, use_food

    @property
    def reward_events(self) -> Dict[str, float]:
        return {
            # Phase-based per-frame rewards (state classes)
            "survival_day": 0.01,
            "survival_night": 0.02,
            "mining": 0.0,
            "building": 0.0,
            "combat": 0.0,
            "inventory": 0.0,
            "creative": 0.005,
            # Resource gathering
            "collect_resource": 0.5,
            "harvest_crop": 0.3,
            "craft_item": 2.0,
            "upgrade_tool": 3.0,
            # Building
            "place_block": 0.2,
            # Combat
            "defeat_mob": 2.0,
            "defeat_boss": 10.0,
            "take_damage": -0.5,
            # Survival
            "survive_night": 5.0,
            "starve": -1.0,
            # Exploration
            "explore_new_area": 1.0,
            "tame_pet": 5.0,
            # Baseline
            "normal": 0.01,
            "other": -0.002,
        }

    @property
    def terminal_events(self) -> List[str]:
        # Sandbox game: no hard terminal state (respawn on death)
        return []

    @property
    def touch_mapping(self) -> Dict[str, TouchAction]:
        """Touch coordinates for 1920x1080 landscape resolution."""
        move_center = (300, 850)
        move_radius = 150

        mapping: Dict[str, TouchAction] = {}

        # Movement (left joystick)
        offsets = {
            "forward": (0, -move_radius),
            "backward": (0, move_radius),
            "left": (-move_radius, 0),
            "right": (move_radius, 0),
            "forward_left": (-move_radius * 2 // 3, -move_radius * 2 // 3),
            "forward_right": (move_radius * 2 // 3, -move_radius * 2 // 3),
            "backward_left": (-move_radius * 2 // 3, move_radius * 2 // 3),
            "backward_right": (move_radius * 2 // 3, move_radius * 2 // 3),
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
            "jump": (1650, 800),
            "break_block": (1750, 900),    # fist icon
            "place_block": (1600, 900),    # hand icon
            "use_food": (1500, 850),
            "switch_slot_1": (750, 980),
            "switch_slot_2": (850, 980),
            "open_inventory": (1050, 980),
        }

        for name, coord in button_coords.items():
            touch_type = "tap"
            duration = 100
            if name == "break_block":
                duration = 500  # long press to break
            mapping[name] = TouchAction(
                type=touch_type,
                coords=coord,
                duration_ms=duration,
            )

        # Dynamic "look" action: swipe to rotate camera
        screen_cx, screen_cy = 960, 540  # center of 1920x1080
        mapping["look"] = TouchAction(
            type="look",
            coords=(screen_cx, screen_cy, 400),
            duration_ms=50,
            param_keys=("look_dx", "look_dy"),
        )

        return mapping

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
