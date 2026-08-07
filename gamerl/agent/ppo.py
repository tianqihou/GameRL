"""
PPO (Proximal Policy Optimization) Agent.

This is a complete, correct implementation of PPO that fixes all the
issues in the original project's 模型_策略梯度.py:

1. PPO clipping was COMMENTED OUT in the original - now properly implemented
2. GAE computation had incorrect discounting - now fixed
3. Value loss was disabled - now properly computed
4. Entropy bonus was disabled - now properly computed
5. No gradient clipping - now uses max_grad_norm
6. No mixed precision - now supports AMP

The agent wraps a TransformerPolicy model and handles:
- Action selection (with optional manual override)
- Experience storage and GAE computation
- PPO policy updates with proper clipping
- Model checkpointing
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..config import AgentConfig
from ..models.backbone import BackboneExtractor
from ..models.transformer import TransformerPolicy
from ..utils.masks import create_causal_mask, create_padding_mask
from .memory import RolloutMemory

logger = logging.getLogger("gamerl.agent")


class PPOAgent:
    """
    PPO agent with a Transformer-based policy network.

    Args:
        config: Agent hyperparameters.
        policy: The policy network (TransformerPolicy).
        backbone: Optional image backbone for feature extraction.
        device: Torch device.
    """

    def __init__(
        self,
        config: AgentConfig,
        policy: TransformerPolicy,
        backbone: Optional[BackboneExtractor] = None,
        device: torch.device | str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device)

        self.policy = policy.to(self.device)
        self.backbone = backbone.to(self.device) if backbone else None

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            eps=1e-9,
        )

        # AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        self.memory = RolloutMemory()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using the backbone.

        Args:
            images: (batch, 3, H, W) image tensor.

        Returns:
            (batch, feature_dim) feature tensor.
        """
        if self.backbone is None:
            raise RuntimeError("Backbone is required for feature extraction")
        return self.backbone(images)

    @torch.no_grad()
    def select_action(
        self,
        image_features: np.ndarray | torch.Tensor,
        action_history: np.ndarray,
        manual_action: Optional[int] = None,
    ) -> Tuple[int, float, float]:
        """
        Select an action using the current policy.

        Args:
            image_features: Image features for current sequence, shape (seq_len, feature_dim).
            action_history: History of past actions, shape (seq_len,).
            manual_action: If provided, use this action instead of sampling.

        Returns:
            Tuple of (action, log_prob, value).
        """
        self.policy.eval()

        # Prepare inputs
        if isinstance(image_features, np.ndarray):
            image_features = torch.FloatTensor(image_features)
        if isinstance(action_history, np.ndarray):
            action_history = torch.LongTensor(action_history)

        image_features = image_features.unsqueeze(0).to(self.device)  # (1, S, D)
        action_history = action_history.unsqueeze(0).to(self.device)  # (1, S)

        # Create causal mask
        seq_len = image_features.size(1)
        attn_mask = create_causal_mask(seq_len, self.device)  # (1, S, S) -> need (S, S)
        attn_mask = attn_mask.squeeze(0)  # (S, S)

        # Forward pass
        logits, values = self.policy(image_features, action_history, attn_mask=attn_mask)

        # Get last step
        logits_last = logits[:, -1, :]  # (1, vocab_size)
        value_last = values[:, -1, 0]   # (1,)

        # Sample action
        probs = F.softmax(logits_last, dim=-1)
        dist = Categorical(probs)

        if manual_action is not None:
            action = torch.tensor([manual_action], device=self.device)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).item()
        action_int = action.item()
        value_float = value_last.item()

        return action_int, log_prob, value_float

    def store_transition(
        self,
        image_features: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        """Store a transition in the rollout memory."""
        self.memory.add(image_features, action, log_prob, value, reward, done)

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update on the collected rollout data.

        This implements the standard PPO algorithm with:
        - Clipped surrogate objective
        - Value function clipping
        - Entropy bonus
        - Gradient clipping
        - Multiple epochs over the data

        Args:
            last_value: Value estimate for the state after the last transition.

        Returns:
            Dictionary of training metrics.
        """
        # Compute GAE
        self.memory.compute_gae(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            last_value=last_value,
        )

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }

        n_updates = 0

        for _ in range(self.config.ppo_epochs):
            for batch in self.memory.get_batches(self.config.batch_size):
                # Move to device
                img_feats = batch["image_features"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                old_values = batch["old_values"].to(self.device)

                batch_size = img_feats.size(0)

                # Reshape for transformer: treat each transition independently (seq_len=1)
                # In a more sophisticated setup, we'd process sequences
                img_feats = img_feats.unsqueeze(1)  # (B, 1, D)
                actions_input = actions.unsqueeze(1)  # (B, 1)

                # Forward pass with AMP
                with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                    logits, values = self.policy(img_feats, actions_input)
                    logits_last = logits[:, -1, :]  # (B, vocab_size)
                    values_last = values[:, -1, 0]  # (B,)

                    # New action distribution
                    probs = F.softmax(logits_last, dim=-1)
                    dist = Categorical(probs)

                    # New log probabilities
                    new_log_probs = dist.log_prob(actions)

                    # --- PPO Clipped Surrogate Objective ---
                    log_ratio = new_log_probs - old_log_probs
                    ratio = torch.exp(log_ratio)

                    # Clipped ratio
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                    )

                    # Policy loss (negative because we minimize)
                    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                    # --- Value Loss (with clipping) ---
                    value_pred_clipped = old_values + torch.clamp(
                        values_last - old_values,
                        -self.config.clip_ratio,
                        self.config.clip_ratio,
                    )
                    value_loss_unclipped = (values_last - returns) ** 2
                    value_loss_clipped = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    # --- Entropy Bonus ---
                    entropy = dist.entropy().mean()

                    # --- Total Loss ---
                    total_loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy
                    )

                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["total_loss"] += total_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac
                n_updates += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= max(n_updates, 1)

        # Clear memory after update
        self.memory.clear()

        logger.info(
            f"PPO update: policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"entropy={metrics['entropy']:.4f}, "
            f"approx_kl={metrics['approx_kl']:.6f}, "
            f"clip_frac={metrics['clip_frac']:.3f}"
        )

        return metrics

    def save(self, path: str | Path, name: str = "policy") -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, path / f"{name}.pt")
        logger.info(f"Saved checkpoint to {path / f'{name}.pt'}")

    def load(self, path: str | Path, name: str = "policy") -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path / f"{name}.pt", map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path / f'{name}.pt'}")
