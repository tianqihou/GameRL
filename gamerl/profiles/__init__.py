"""
Game profile registry and factory.

Each game provides a GameProfile that defines its action space, state
classes, touch mappings, screen regions, and reward configuration.
The generic RL infrastructure (PPO, Transformer, backbone, etc.) works
with any profile, enabling the same codebase to support multiple games.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import GameProfile
from .honor_of_kings import HonorOfKingsProfile
from .peacekeeper import PeacekeeperEliteProfile
from .genshin import GenshinImpactProfile
from .mini_world import MiniWorldProfile
from .roco_kingdom import RocoKingdomProfile

# Registry of available game profiles
PROFILES: Dict[str, Type[GameProfile]] = {
    "honor_of_kings": HonorOfKingsProfile,
    "peacekeeper": PeacekeeperEliteProfile,
    "genshin": GenshinImpactProfile,
    "mini_world": MiniWorldProfile,
    "roco_kingdom": RocoKingdomProfile,
}

# Human-readable aliases
ALIASES: Dict[str, str] = {
    "hok": "honor_of_kings",
    "wzry": "honor_of_kings",
    "王者荣耀": "honor_of_kings",
    "和平精英": "peacekeeper",
    "pubg_mobile": "peacekeeper",
    "原神": "genshin",
    "gi": "genshin",
    "迷你世界": "mini_world",
    "mw": "mini_world",
    "miniworld": "mini_world",
    "洛克王国": "roco_kingdom",
    "洛克王国世界": "roco_kingdom",
    "rkc": "roco_kingdom",
}


def get_profile(name: str, **kwargs) -> GameProfile:
    """
    Get a game profile by name.

    Args:
        name: Profile name (e.g., "honor_of_kings") or alias (e.g., "wzry").
        **kwargs: Additional arguments passed to the profile constructor.

    Returns:
        A GameProfile instance.

    Raises:
        ValueError: If the profile name is not recognized.
    """
    key = ALIASES.get(name, name)
    if key not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(
            f"Unknown game profile '{name}'. Available: {available}"
        )
    return PROFILES[key](**kwargs)


def list_profiles() -> Dict[str, str]:
    """List all available profiles with their display names."""
    result = {}
    for key, cls in PROFILES.items():
        result[key] = cls.display_name
    return result


__all__ = [
    "GameProfile",
    "HonorOfKingsProfile",
    "PeacekeeperEliteProfile",
    "GenshinImpactProfile",
    "MiniWorldProfile",
    "RocoKingdomProfile",
    "PROFILES",
    "ALIASES",
    "get_profile",
    "list_profiles",
]
