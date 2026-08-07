"""
Structured game state builder.

Converts raw YOLO detections into a fixed-size structured state vector
that can be fed to the policy network. This is more sample-efficient
than raw CNN features because the state representation is compact and
meaningful.

Example structured state for a MOBA game:

    StructuredState(
        player_hp=0.75,
        player_pos=(0.3, 0.7),
        enemies=[
            EnemyState(pos=(0.5, 0.4), hp=0.3, distance=0.2),
            ...
        ],
        skills=[1, 0, 1],  # 1=available, 0=cooldown
        towers_visible=[(0.1, 0.9, 'enemy'), ...],
        minions_nearby=[(0.2, 0.8), ...],
    )

The state is flattened to a fixed-size vector for the Transformer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection, DetectionResult

logger = logging.getLogger("gamerl.vision")


@dataclass
class EnemyState:
    """State of a single enemy unit."""

    pos: Tuple[float, float]  # Normalized (0-1) position
    hp: float = 1.0  # Normalized HP (0-1), 1.0 if unknown
    distance: float = 0.0  # Normalized distance to player (0-1)
    confidence: float = 0.0  # Detection confidence


@dataclass
class TowerState:
    """State of a tower."""

    pos: Tuple[float, float]
    team: str = "enemy"  # "ally" | "enemy" | "unknown"
    hp: float = 1.0


@dataclass
class StructuredState:
    """
    Structured representation of the game state at a point in time.

    All positions are normalized to [0, 1] relative to the screen.
    """

    player_hp: float = 1.0
    player_pos: Tuple[float, float] = (0.5, 0.5)
    enemies: List[EnemyState] = field(default_factory=list)
    skills_available: List[int] = field(default_factory=list)  # 1=available, 0=cooldown
    towers: List[TowerState] = field(default_factory=list)
    minions: List[Tuple[float, float]] = field(default_factory=list)  # positions only
    gold: float = 0.0  # Normalized gold estimate
    timestamp: float = 0.0  # Game time in seconds

    # Metadata
    detection_time_ms: float = 0.0
    num_detections: int = 0

    def to_vector(
        self,
        max_enemies: int = 5,
        max_towers: int = 6,
        max_minions: int = 10,
        num_skills: int = 3,
    ) -> np.ndarray:
        """
        Flatten to a fixed-size vector for the policy network.

        Layout (total dims):
            player_hp          : 1
            player_pos         : 2
            enemies (padded)   : max_enemies * 4  (x, y, hp, distance)
            skills_available   : num_skills
            towers (padded)    : max_towers * 3  (x, y, team_id)
            minions (padded)   : max_minions * 2  (x, y)
            gold               : 1

        Total = 1 + 2 + max_enemies*4 + num_skills + max_towers*3 + max_minions*2 + 1

        Args:
            max_enemies: Maximum number of enemies to encode (pad if fewer).
            max_towers: Maximum number of towers to encode.
            max_minions: Maximum number of minions to encode.
            num_skills: Number of skill slots.

        Returns:
            1D float32 array of shape (total_dims,).
        """
        parts: List[np.ndarray] = []

        # Player state
        parts.append(np.array([self.player_hp], dtype=np.float32))
        parts.append(np.array(self.player_pos, dtype=np.float32))

        # Enemies (padded)
        enemy_data = np.zeros((max_enemies, 4), dtype=np.float32)
        for i, enemy in enumerate(self.enemies[:max_enemies]):
            enemy_data[i] = [enemy.pos[0], enemy.pos[1], enemy.hp, enemy.distance]
        parts.append(enemy_data.flatten())

        # Skills
        skill_data = np.zeros(num_skills, dtype=np.float32)
        for i, s in enumerate(self.skills_available[:num_skills]):
            skill_data[i] = s
        parts.append(skill_data)

        # Towers (padded)
        tower_data = np.zeros((max_towers, 3), dtype=np.float32)
        team_map = {"ally": 0.0, "enemy": 1.0, "unknown": 0.5}
        for i, tower in enumerate(self.towers[:max_towers]):
            tower_data[i] = [tower.pos[0], tower.pos[1], team_map.get(tower.team, 0.5)]
        parts.append(tower_data.flatten())

        # Minions (padded)
        minion_data = np.zeros((max_minions, 2), dtype=np.float32)
        for i, pos in enumerate(self.minions[:max_minions]):
            minion_data[i] = [pos[0], pos[1]]
        parts.append(minion_data.flatten())

        # Gold
        parts.append(np.array([self.gold], dtype=np.float32))

        return np.concatenate(parts)

    @staticmethod
    def vector_dim(
        max_enemies: int = 5,
        max_towers: int = 6,
        max_minions: int = 10,
        num_skills: int = 3,
    ) -> int:
        """Calculate the dimension of the state vector."""
        return 1 + 2 + max_enemies * 4 + num_skills + max_towers * 3 + max_minions * 2 + 1

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging/debugging."""
        return {
            "player_hp": self.player_hp,
            "player_pos": list(self.player_pos),
            "enemies": [
                {"pos": list(e.pos), "hp": e.hp, "distance": e.distance}
                for e in self.enemies
            ],
            "skills_available": self.skills_available,
            "towers": [
                {"pos": list(t.pos), "team": t.team, "hp": t.hp} for t in self.towers
            ],
            "minions": [list(m) for m in self.minions],
            "gold": self.gold,
            "detection_time_ms": self.detection_time_ms,
            "num_detections": self.num_detections,
        }


class GameStateBuilder:
    """
    Convert raw detections into a structured game state.

    This class bridges the gap between YOLO bounding boxes and the
    compact state representation that the policy network consumes.

    Args:
        class_names: List of detection class names (must match detector).
        player_class: Class name for the player's hero.
        enemy_class: Class name for enemy heroes.
        minion_class: Class name for minions.
        tower_class: Class name for towers.
        hp_bar_class: Class name for HP bars.
        skill_class: Class name for skill buttons.
        screen_resolution: (width, height) of the game screen.
        max_enemies: Max enemies to encode in state vector.
        max_towers: Max towers to encode.
        max_minions: Max minions to encode.
        num_skills: Number of skill slots.
    """

    def __init__(
        self,
        class_names: List[str],
        player_class: str = "hero",
        enemy_class: str = "enemy",
        minion_class: str = "minion",
        tower_class: str = "tower",
        hp_bar_class: str = "hp_bar",
        skill_class: str = "skill_button",
        screen_resolution: Tuple[int, int] = (1080, 2160),
        max_enemies: int = 5,
        max_towers: int = 6,
        max_minions: int = 10,
        num_skills: int = 3,
    ):
        self.class_names = class_names
        self.player_class = player_class
        self.enemy_class = enemy_class
        self.minion_class = minion_class
        self.tower_class = tower_class
        self.hp_bar_class = hp_bar_class
        self.skill_class = skill_class
        self.screen_w, self.screen_h = screen_resolution
        self.max_enemies = max_enemies
        self.max_towers = max_towers
        self.max_minions = max_minions
        self.num_skills = num_skills

        # Previous player position for distance computation
        self._prev_player_pos: Optional[Tuple[float, float]] = None

    def build(self, result: DetectionResult) -> StructuredState:
        """
        Build a structured state from detection results.

        Args:
            result: Detection results from the detector.

        Returns:
            StructuredState with parsed game information.
        """
        h, w = result.frame_shape
        if h == 0 or w == 0:
            h, w = self.screen_h, self.screen_w

        # Find player (highest confidence player detection, or center if none)
        player_dets = result.filter_by_class(self.player_class)
        if player_dets:
            player = max(player_dets, key=lambda d: d.confidence)
            player_pos = self._normalize_pos(player.center, w, h)
        else:
            # Fallback: assume player is at bottom-center
            player_pos = (0.5, 0.75)

        # Extract player HP from HP bar detection near player
        player_hp = self._estimate_hp(player_dets, result, w, h)

        # Find enemies
        enemy_dets = result.filter_by_class(self.enemy_class)
        enemies = self._build_enemies(enemy_dets, player_pos, w, h)

        # Find minions
        minion_dets = result.filter_by_class(self.minion_class)
        minions = self._build_minions(minion_dets, player_pos, w, h)

        # Find towers
        tower_dets = result.filter_by_class(self.tower_class)
        towers = self._build_towers(tower_dets, w, h)

        # Extract skill availability
        skill_dets = result.filter_by_class(self.skill_class)
        skills = self._extract_skills(skill_dets)

        state = StructuredState(
            player_hp=player_hp,
            player_pos=player_pos,
            enemies=enemies,
            skills_available=skills,
            towers=towers,
            minions=minions,
            detection_time_ms=result.inference_time_ms,
            num_detections=len(result.detections),
        )

        self._prev_player_pos = player_pos
        return state

    def build_vector(self, result: DetectionResult) -> np.ndarray:
        """Convenience: build state and flatten to vector in one call."""
        state = self.build(result)
        return state.to_vector(
            max_enemies=self.max_enemies,
            max_towers=self.max_towers,
            max_minions=self.max_minions,
            num_skills=self.num_skills,
        )

    def _normalize_pos(
        self,
        pos: Tuple[float, float],
        w: int,
        h: int,
    ) -> Tuple[float, float]:
        """Normalize pixel position to [0, 1]."""
        return (pos[0] / max(w, 1), pos[1] / max(h, 1))

    def _estimate_hp(
        self,
        player_dets: List[Detection],
        result: DetectionResult,
        w: int,
        h: int,
    ) -> float:
        """
        Estimate player HP from HP bar detection.

        Uses the fill ratio of the HP bar closest to the player.
        Falls back to 1.0 if no HP bar is detected.
        """
        if not player_dets:
            return 1.0

        player = max(player_dets, key=lambda d: d.confidence)
        hp_bars = result.filter_by_class(self.hp_bar_class)

        if not hp_bars:
            return 1.0

        # Find HP bar closest to player
        player_center = player.center
        closest_hp = min(hp_bars, key=lambda d: self._dist(d.center, player_center))

        # Estimate HP from HP bar aspect ratio
        # A full HP bar has a certain width; the colored portion indicates HP
        # This is a simplified heuristic; real implementation would use color analysis
        return 1.0  # Placeholder; real HP estimation requires color analysis

    def _build_enemies(
        self,
        enemy_dets: List[Detection],
        player_pos: Tuple[float, float],
        w: int,
        h: int,
    ) -> List[EnemyState]:
        """Build enemy state list from detections."""
        # Sort by distance to player (closest first)
        player_px = (player_pos[0] * w, player_pos[1] * h)

        def enemy_key(d: Detection) -> float:
            return self._dist(d.center, player_px)

        sorted_enemies = sorted(enemy_dets, key=enemy_key)

        enemies = []
        for det in sorted_enemies[: self.max_enemies]:
            pos = self._normalize_pos(det.center, w, h)
            dist = self._normalized_distance(pos, player_pos)
            enemies.append(
                EnemyState(
                    pos=pos,
                    hp=1.0,  # Would need color analysis for enemy HP
                    distance=dist,
                    confidence=det.confidence,
                )
            )

        return enemies

    def _build_minions(
        self,
        minion_dets: List[Detection],
        player_pos: Tuple[float, float],
        w: int,
        h: int,
    ) -> List[Tuple[float, float]]:
        """Build minion position list (closest first)."""
        player_px = (player_pos[0] * w, player_pos[1] * h)

        sorted_minions = sorted(
            minion_dets, key=lambda d: self._dist(d.center, player_px)
        )

        return [
            self._normalize_pos(d.center, w, h)
            for d in sorted_minions[: self.max_minions]
        ]

    def _build_towers(
        self,
        tower_dets: List[Detection],
        w: int,
        h: int,
    ) -> List[TowerState]:
        """Build tower state list from detections."""
        towers = []
        for det in tower_dets[: self.max_towers]:
            pos = self._normalize_pos(det.center, w, h)
            # Tower team is hard to determine from detection alone
            # In practice, you'd use position heuristics or color analysis
            # Towers on the bottom half are typically enemy, top half are ally
            team = "enemy" if pos[1] < 0.5 else "ally"
            towers.append(TowerState(pos=pos, team=team))

        return towers

    def _extract_skills(self, skill_dets: List[Detection]) -> List[int]:
        """
        Extract skill availability from skill button detections.

        A skill is "available" if its button is detected with high confidence
        (cooldown skills are typically dimmed/grayed and may not be detected).
        """
        skills = [0] * self.num_skills
        for i, det in enumerate(skill_dets[: self.num_skills]):
            skills[i] = 1 if det.confidence > 0.5 else 0
        return skills

    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _normalized_distance(
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        """Normalized distance between two [0,1] positions."""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
