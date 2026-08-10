"""
Rollout memory for PPO training.

Stores transitions during gameplay and provides batched sampling for training.
Replaces the PPO_数据集 class from the original project with a memory-efficient
implementation that doesn't pre-allocate huge numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class Transition:
    """A single state-action transition."""

    image_features: np.ndarray  # (feature_dim,)
    action: int
    action_log_prob: float
    value: float
    reward: float
    done: bool
    # Hybrid action space (optional)
    continuous_params: np.ndarray | None = None  # (continuous_dim,)
    continuous_log_prob: float = 0.0


class RolloutMemory:
    """
    Memory buffer for storing PPO rollout data.

    Efficiently stores transitions and provides methods for:
    - Computing GAE (Generalized Advantage Estimation)
    - Batched sampling for PPO updates
    - Serialization to/from disk

    Unlike the original PPO_数据集 which pre-allocated a 73MB numpy array,
    this uses dynamic lists and only converts to tensors when needed.
    """

    def __init__(self, max_size: int = 32768):
        """
        Initialize the rollout memory.

        Args:
            max_size: Maximum number of transitions to store.

            PPO benefits from longer rollouts in sparse-reward games; default is increased to 32768.
        """
        self.max_size = max_size
        self.image_features: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        # Hybrid action data
        self.continuous_params: List[np.ndarray | None] = []
        self.continuous_log_probs: List[float] = []

        # Computed during GAE
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        image_features: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        continuous_params: np.ndarray | None = None,
        continuous_log_prob: float = 0.0,
    ) -> None:
        """Add a transition to the buffer."""
        if len(self.actions) >= self.max_size:
            # Remove oldest entry (sliding window)
            self.image_features.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.continuous_params.pop(0)
            self.continuous_log_probs.pop(0)

        self.image_features.append(image_features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.continuous_params.append(continuous_params)
        self.continuous_log_probs.append(continuous_log_prob)

    def compute_gae(
        self,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        """
        Compute Generalized Advantage Estimation (GAE).

        This fixes the buggy GAE computation in the original project,
        which had incorrect discounting and didn't properly handle
        episode boundaries.

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            last_value: Value estimate for the state after the last transition.
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        # Convert values to array with last_value appended
        values = np.array(self.values + [last_value], dtype=np.float32)

        for t in reversed(range(n)):
            # If episode ended, no bootstrap
            delta = self.rewards[t] + gamma * values[t + 1] * (1.0 - self.dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - self.dones[t]) * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Yield batches of data for PPO training.

        Normalizes advantages for training stability.

        Args:
            batch_size: Number of transitions per batch.

        Yields:
            Dictionary with keys: image_features, actions, old_log_probs,
            advantages, returns, values.
            When hybrid data is present, also includes:
            continuous_params, old_continuous_log_probs.
        """
        assert self.advantages is not None, "Must call compute_gae() first"

        n = len(self.actions)
        indices = np.random.permutation(n)

        # Normalize advantages
        adv_normalized = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Check if hybrid data is present
        has_continuous = self.continuous_params and self.continuous_params[0] is not None

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]

            batch: Dict[str, torch.Tensor] = {
                "image_features": torch.FloatTensor(
                    np.stack([self.image_features[i] for i in batch_idx])
                ),
                "actions": torch.LongTensor([self.actions[i] for i in batch_idx]),
                "old_log_probs": torch.FloatTensor([self.log_probs[i] for i in batch_idx]),
                "advantages": torch.FloatTensor(adv_normalized[batch_idx]),
                "returns": torch.FloatTensor(self.returns[batch_idx]),
                "old_values": torch.FloatTensor([self.values[i] for i in batch_idx]),
            }

            if has_continuous:
                batch["continuous_params"] = torch.FloatTensor(
                    np.stack([self.continuous_params[i] for i in batch_idx])
                )
                batch["old_continuous_log_probs"] = torch.FloatTensor(
                    [self.continuous_log_probs[i] for i in batch_idx]
                )

            yield batch

    def clear(self) -> None:
        """Clear all stored data."""
        self.image_features.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.continuous_params.clear()
        self.continuous_log_probs.clear()
        self.advantages = None
        self.returns = None

    def save(self, path: str) -> None:
        """Save rollout data to disk."""
        data = {
            "image_features": np.stack(self.image_features),
            "actions": np.array(self.actions),
            "log_probs": np.array(self.log_probs),
            "values": np.array(self.values),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
        }
        # Serialize continuous params when present (object array preserves None)
        if self.continuous_params:
            data["continuous_params"] = np.array(self.continuous_params, dtype=object)
            data["continuous_log_probs"] = np.array(self.continuous_log_probs)
        np.savez_compressed(path, **data)

    def load(self, path: str) -> None:
        """Load rollout data from disk."""
        data = np.load(path, allow_pickle=True)
        self.image_features = list(data["image_features"])
        self.actions = list(data["actions"])
        self.log_probs = list(data["log_probs"])
        self.values = list(data["values"])
        self.rewards = list(data["rewards"])
        self.dones = list(data["dones"])
        # Restore continuous params (may be absent in older saves)
        if "continuous_params" in data:
            self.continuous_params = list(data["continuous_params"])
            self.continuous_log_probs = list(data["continuous_log_probs"])
        else:
            self.continuous_params = [None] * len(self.actions)
            self.continuous_log_probs = [0.0] * len(self.actions)
