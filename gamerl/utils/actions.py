"""
Action space definition for game AI.

The action space is a combination of movement direction and action type,
encoded as a single discrete token. This mirrors the original project's
"方向_动作" vocabulary but with a clean, documented API.

The ActionSpace class is game-agnostic and works with string labels.
Game-specific enums (Movement, ActionType) are provided for backward
compatibility with the Honor of Kings profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import json
from pathlib import Path


class Movement(Enum):
    """Movement directions (joystick) - Honor of Kings specific."""

    UP = "上移"
    DOWN = "下移"
    LEFT = "左移"
    RIGHT = "右移"
    UP_LEFT = "左上移"
    UP_RIGHT = "右上移"
    DOWN_LEFT = "左下移"
    DOWN_RIGHT = "右下移"
    STOP = "移动停"
    NONE = "无移动"


class ActionType(Enum):
    """Action types (buttons) - Honor of Kings specific."""

    ATTACK = "攻击"
    LAST_HIT = "补刀"
    PUSH_TOWER = "推塔"
    SKILL_1 = "一技能"
    SKILL_2 = "二技能"
    SKILL_3 = "三技能"
    SUMMONER = "召唤师技能"
    RECALL = "回城"
    SIGNAL_ATTACK = "发起进攻"
    SIGNAL_RETREAT = "发起撤退"
    SIGNAL_GATHER = "发起集合"
    NO_ACTION = "无动作"
    HEAL = "恢复"


# Special tokens (backward compatibility)
PAD_TOKEN = -1
BOS_TOKEN = 128  # Honor of Kings BOS - within the 130-token vocab
NUM_SPECIAL_TOKENS = 0


@dataclass
class ActionSpace:
    """
    Discrete action space combining movement and action type.

    Total actions = len(movements) * len(actions)
    Works with any game's string-based action labels.

    Args:
        movements: List of movement direction labels (strings).
        actions: List of action/button labels (strings).
        bos_token: Token ID for beginning-of-sequence (within vocab range).
    """

    movements: List[str] = field(default_factory=lambda: [m.value for m in Movement])
    actions: List[str] = field(default_factory=lambda: [a.value for a in ActionType])
    bos_token: int = BOS_TOKEN

    @property
    def size(self) -> int:
        """Total number of action tokens (excluding special)."""
        return len(self.movements) * len(self.actions)

    @property
    def vocab_size(self) -> int:
        """Full vocabulary size including special tokens."""
        return self.size + NUM_SPECIAL_TOKENS

    def encode(self, movement: str, action: str) -> int:
        """Encode a (movement, action) pair into a single token."""
        m_idx = self.movements.index(movement)
        a_idx = self.actions.index(action)
        return m_idx * len(self.actions) + a_idx

    def decode(self, token: int) -> Tuple[str, str]:
        """Decode a token back into (movement, action) as strings."""
        m_idx = token // len(self.actions)
        a_idx = token % len(self.actions)
        return self.movements[m_idx], self.actions[a_idx]

    def decode_to_str(self, token: int) -> str:
        """Decode a token to the original '方向_动作' string format."""
        if token == self.bos_token:
            m, a = self.decode(token)
            return f"{m}_{a}"
        m, a = self.decode(token)
        return f"{m}_{a}"

    def build_vocab(self) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Build word-to-index and index-to-word mappings.

        Returns:
            Tuple of (word_to_idx, idx_to_word) dictionaries.
        """
        word_to_idx: Dict[str, int] = {}
        idx_to_word: Dict[str, str] = {}
        for m in self.movements:
            for a in self.actions:
                key = f"{m}_{a}"
                idx = self.encode(m, a)
                word_to_idx[key] = idx
                idx_to_word[str(idx)] = key
        return word_to_idx, idx_to_word

    def save_vocab(self, path: str | Path) -> None:
        """Save vocabulary to JSON files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        word_to_idx, idx_to_word = self.build_vocab()
        with open(path / "action_vocab.json", "w", encoding="utf-8") as f:
            json.dump(word_to_idx, f, ensure_ascii=False, indent=2)
        with open(path / "action_idx.json", "w", encoding="utf-8") as f:
            json.dump(idx_to_word, f, ensure_ascii=False, indent=2)


def default_action_space() -> ActionSpace:
    """Create the default action space for Honor of Kings."""
    return ActionSpace()
