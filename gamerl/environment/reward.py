"""
Reward shaping for game-specific reward computation.

Bridges the gap between:
1. StateJudgmentModel (classifies game events from screenshots)
2. GameProfile.reward_events (maps event names to reward weights)
3. StrategicDirective.reward_weights (runtime strategic adjustments)

The RewardShaper computes a scalar reward from each state transition
and determines whether the episode should terminate.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("gamerl.environment.reward")


class RewardShaper:
    """
    Computes scalar rewards from game state transitions.

    Pipeline:
    1. Classify the current frame into an event category (via StateJudgmentModel
       or a custom callback).
    2. Look up the reward weight for that event in reward_events.
    3. Apply strategic directive reward weights (optional runtime shaping).
    4. Check if the event is terminal (episode should end).

    Args:
        reward_events: Dict mapping event names to reward weights.
            Must include 'normal' and 'other' keys at minimum.
        state_classes: List of event category names (for StateJudgmentModel).
            The classifier outputs an index into this list.
        terminal_events: Event names that signal episode termination.
        state_model: Optional StateJudgmentModel for event classification.
            If None, a custom event_callback must be provided, or all
            events default to 'normal'.
        event_callback: Optional callable that takes (prev_features,
            curr_features, action) and returns an event name string.
            Overrides state_model if both are provided.
        default_event: Event name to use when no classification is available.
            Defaults to 'normal'.
    """

    def __init__(
        self,
        reward_events: Dict[str, float],
        state_classes: Optional[List[str]] = None,
        terminal_events: Optional[List[str]] = None,
        state_model=None,
        event_callback=None,
        default_event: str = "normal",
    ):
        self.reward_events = reward_events
        self.state_classes = state_classes or list(reward_events.keys())
        self.terminal_events = terminal_events or []
        self.state_model = state_model
        self.event_callback = event_callback
        self.default_event = default_event

        # Build event → index mapping for state_model output
        self._event_to_idx = {name: i for i, name in enumerate(self.state_classes)}

        # Stats
        self._event_counts: Dict[str, int] = {}
        self._total_reward = 0.0
        self._step_count = 0

    def compute_reward(
        self,
        curr_features: Optional[np.ndarray] = None,
        action: int = 0,
        action_seq: Optional[np.ndarray] = None,
        strategic_weights: Optional[Dict[str, float]] = None,
        prev_features: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool, str]:
        """
        Compute reward and done flag for a single state transition.

        Args:
            curr_features: Current frame image features (feature_dim,) or
                (seq_len, feature_dim). Used for state_model classification.
            action: Action token taken this step.
            action_seq: Full action history (for state_model). If None,
                a single-element tensor with `action` is used.
            strategic_weights: Optional dict from StrategicDirective to
                add bonus/penalty weights at runtime.
            prev_features: Previous frame features (for event_callback).

        Returns:
            Tuple of (reward, done, event_name).
        """
        # Step 1: Classify the event
        event = self._classify_event(
            curr_features=curr_features,
            action=action,
            action_seq=action_seq,
            prev_features=prev_features,
        )

        # Step 2: Base reward from event
        reward = self.reward_events.get(event, self.reward_events.get("other", 0.0))

        # Step 3: Apply strategic directive weights
        if strategic_weights:
            for key, weight in strategic_weights.items():
                # Apply if the key matches the event exactly,
                # or if the key is a substring of the event (e.g., "kill" matches "kill_hero")
                if key == event or key in event:
                    reward += weight

        # Step 4: Check terminal
        done = event in self.terminal_events

        # Update stats
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        self._total_reward += reward
        self._step_count += 1

        return reward, done, event

    def _classify_event(
        self,
        curr_features: Optional[np.ndarray] = None,
        action: int = 0,
        action_seq: Optional[np.ndarray] = None,
        prev_features: Optional[np.ndarray] = None,
    ) -> str:
        """Classify the current state into an event category name."""
        # Priority 1: Custom callback
        if self.event_callback is not None:
            try:
                event = self.event_callback(prev_features, curr_features, action)
                if event and event in self.reward_events:
                    return event
                if event:
                    return event  # Allow custom events not in state_classes
            except Exception as e:
                logger.warning(f"Event callback error: {e}")

        # Priority 2: StateJudgmentModel
        if self.state_model is not None and curr_features is not None:
            try:
                event = self._classify_with_model(curr_features, action_seq, action)
                if event:
                    return event
            except Exception as e:
                logger.warning(f"State model classification error: {e}")

        # Fallback: default event
        return self.default_event

    def _classify_with_model(
        self,
        features: np.ndarray,
        action_seq: Optional[np.ndarray],
        action: int,
    ) -> Optional[str]:
        """Use StateJudgmentModel to classify the current frame."""
        # Prepare input tensor
        if features.ndim == 1:
            features = features[np.newaxis, :]  # (1, feature_dim)

        tensor = torch.from_numpy(features).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1, seq_len, feature_dim)

        if torch.cuda.is_available():
            tensor = tensor.cuda()

        # Prepare action sequence
        if action_seq is not None:
            act_tensor = torch.from_numpy(action_seq).long().unsqueeze(0)
        else:
            act_tensor = torch.tensor([[action]], dtype=torch.long)

        if torch.cuda.is_available():
            act_tensor = act_tensor.cuda()

        # Classify (use last timestep)
        with torch.no_grad():
            logits = self.state_model(tensor, act_tensor)  # (1, seq_len, num_classes)
            pred_idx = logits[0, -1].argmax().item()

        if pred_idx < len(self.state_classes):
            return self.state_classes[pred_idx]
        return None

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self._event_counts.clear()
        self._total_reward = 0.0
        self._step_count = 0

    def get_stats(self) -> Dict:
        """Return current statistics."""
        return {
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "mean_reward": self._total_reward / max(self._step_count, 1),
            "event_counts": dict(self._event_counts),
            "event_distribution": {
                k: v / max(self._step_count, 1)
                for k, v in self._event_counts.items()
            },
        }

    @classmethod
    def from_profile(cls, profile, state_model=None, event_callback=None) -> "RewardShaper":
        """
        Create a RewardShaper from a GameProfile.

        Args:
            profile: A GameProfile instance with reward_events defined.
            state_model: Optional StateJudgmentModel.
            event_callback: Optional custom event classification callback.

        Returns:
            Configured RewardShaper instance.
        """
        return cls(
            reward_events=profile.reward_events,
            state_classes=profile.state_classes,
            terminal_events=profile.terminal_events,
            state_model=state_model,
            event_callback=event_callback,
            default_event=profile.reward_events.get("normal", 0.0)
            and "normal"
            or "normal",
        )
