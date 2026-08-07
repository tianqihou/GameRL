"""
Real-time vision pipeline for game state perception.

This module replaces the original project's approach of passing raw CNN features
directly to the policy network. Instead, it uses YOLO object detection to
extract structured game state information:

    Frame → OpenCV preprocess → YOLO detect → StateBuilder → structured vector

The structured state vector includes:
- Player HP and position
- Enemy positions and HP
- Skill availability
- Tower states
- Nearby minions

This is more sample-efficient and interpretable than raw features.
"""

from .preprocess import FramePreprocessor
from .detector import GameDetector, Detection, MockDetector
from .state_builder import GameStateBuilder, StructuredState

__all__ = [
    "FramePreprocessor",
    "GameDetector",
    "Detection",
    "MockDetector",
    "GameStateBuilder",
    "StructuredState",
]
