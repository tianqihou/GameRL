"""
Main training pipeline for the PPO policy network.

Combines data loading, model training, and checkpointing.
Supports:
- Mixed precision training (AMP)
- Gradient clipping
- TensorBoard logging
- Checkpoint resume
- Supervised pretraining + PPO fine-tuning
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from ..agent.ppo import PPOAgent
from ..config import Config
from ..data.dataset import GameSequenceDataset, collate_sequences
from ..models.backbone import BackboneExtractor
from ..models.state_judgment import StateJudgmentModel
from ..models.transformer import TransformerPolicy
from ..utils.actions import ActionSpace, BOS_TOKEN
from ..utils.logging import MetricsLogger
from ..utils.masks import create_causal_mask

logger = logging.getLogger("gamerl.training")


class PolicyTrainer:
    """
    Training pipeline for the policy network.

    Supports two training modes:
    1. Supervised pretraining: Learn from human/AI demonstration data
    2. PPO fine-tuning: Reinforcement learning from game rewards

    Args:
        config: Top-level configuration.
        device: Torch device.
    """

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Build action space
        self.action_space = ActionSpace()

        # Build backbone (frozen, for feature extraction)
        self.backbone = BackboneExtractor(
            backbone_name=config.model.backbone,
            grid_size=config.model.backbone_grid_size,
            pretrained=config.model.pretrained,
            freeze=True,
            use_half=config.model.backbone_half,
        ).to(self.device)

        feature_dim = self.backbone.get_flat_dim()

        # Build policy network
        self.policy = TransformerPolicy(
            feature_dim=feature_dim,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            vocab_size=self.action_space.vocab_size,
            dropout=config.model.dropout,
            max_seq_len=config.model.max_seq_len,
        ).to(self.device)

        # Build PPO agent
        self.agent = PPOAgent(
            config=config.agent,
            policy=self.policy,
            backbone=None,  # Backbone is used separately during data collection
            device=self.device,
        )

        # Metrics logger
        self.metrics = MetricsLogger(config.training.log_dir)

        # State judgment model (for reward computation)
        self.state_model: Optional[StateJudgmentModel] = None

        logger.info(
            f"Policy parameters: {self.policy.count_parameters() / 1e6:.2f}M"
        )

    def load_state_model(self, weights_path: str) -> None:
        """Load the state judgment model for reward computation."""
        feature_dim = self.backbone.get_flat_dim()
        self.state_model = StateJudgmentModel(
            feature_dim=feature_dim,
            d_model=self.config.state_model.d_model,
            n_layers=self.config.state_model.n_layers,
            n_heads=self.config.state_model.n_heads,
            num_classes=self.config.state_model.num_classes,
            dropout=self.config.state_model.dropout,
        ).to(self.device)

        checkpoint = torch.load(weights_path, map_location=self.device)
        if "state_dict" in checkpoint:
            self.state_model.load_state_dict(checkpoint["state_dict"])
        else:
            self.state_model.load_state_dict(checkpoint)
        self.state_model.eval()
        logger.info(f"Loaded state model from {weights_path}")

    def train_supervised(
        self,
        data_dir: str,
        epochs: Optional[int] = None,
    ) -> None:
        """
        Supervised pretraining from demonstration data.

        Trains the policy to predict the next action given the current
        state history, using cross-entropy loss.

        Args:
            data_dir: Directory containing preprocessed .npz files.
            epochs: Number of training epochs (overrides config).
        """
        epochs = epochs or self.config.training.epochs

        dataset = GameSequenceDataset(
            data_dir=data_dir,
            chunk_size=self.config.training.chunk_size,
            stride=self.config.training.stride,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_sequences,
            num_workers=0,
            pin_memory=True,
        )

        # AMP scaler
        scaler = torch.amp.GradScaler("cuda",
            enabled=self.config.training.use_amp and self.device.type == "cuda"
        )

        optimizer = self.agent.optimizer

        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                image_features = batch["image_features"].to(self.device)
                actions = batch["actions"].to(self.device)
                targets = batch["target_actions"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                batch_size, seq_len, _ = image_features.shape

                # Create causal mask
                attn_mask = create_causal_mask(seq_len, self.device).squeeze(0)  # (S, S)

                # Forward pass with AMP
                with torch.amp.autocast("cuda", scaler.is_enabled()):
                    logits, values = self.policy(
                        image_features, actions,
                        attn_mask=attn_mask,
                        key_padding_mask=padding_mask,
                    )

                    # Cross-entropy loss (ignore padding positions)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-1,
                    )

                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.agent.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                # Log metrics
                if global_step % 10 == 0:
                    self.metrics.log_scalar("supervised/loss", loss.item(), global_step)

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.agent.save(
                    self.config.training.weights_dir,
                    name=f"policy_supervised_epoch{epoch + 1}",
                )
                self.agent.save(self.config.training.weights_dir, name="policy_latest")

        self.metrics.close()
        logger.info("Supervised training complete.")

    def train_ppo(
        self,
        env=None,
        episodes: int = 100,
        steps_per_episode: int = 10000,
    ) -> None:
        """
        PPO reinforcement learning training.

        Collects rollout data from the environment and performs
        PPO updates.

        Args:
            env: Game environment instance. If None, only does offline PPO.
            episodes: Number of episodes to train.
            steps_per_episode: Maximum steps per episode.
        """
        if env is None:
            logger.warning("No environment provided. Skipping PPO training.")
            return

        global_step = 0

        for episode in range(episodes):
            logger.info(f"PPO Episode {episode + 1}/{episodes}")

            # Collect rollout
            state = env.reset()
            episode_reward = 0.0

            for step in range(steps_per_episode):
                # Select action
                action, log_prob, value = self.agent.select_action(
                    state.image_features,
                    state.action_history,
                )

                # Take step
                state, reward, done, info = env.step(action)
                episode_reward += reward

                # Store transition
                self.agent.store_transition(
                    image_features=state.image_features[-1],
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                )

                global_step += 1

                if done:
                    break

            # Compute last value for GAE
            with torch.no_grad():
                _, last_value = self.agent.policy(
                    torch.FloatTensor(state.image_features[-1:]).unsqueeze(0).to(self.device),
                    torch.LongTensor(state.action_history[-1:]).unsqueeze(0).to(self.device),
                )
                last_value = last_value.item()

            # PPO update
            metrics = self.agent.update(last_value=last_value)

            # Log metrics
            self.metrics.log_scalar("ppo/policy_loss", metrics["policy_loss"], episode)
            self.metrics.log_scalar("ppo/value_loss", metrics["value_loss"], episode)
            self.metrics.log_scalar("ppo/entropy", metrics["entropy"], episode)
            self.metrics.log_scalar("ppo/episode_reward", episode_reward, episode)
            self.metrics.log_scalar("ppo/approx_kl", metrics["approx_kl"], episode)

            logger.info(
                f"Episode {episode + 1}: reward={episode_reward:.2f}, "
                f"steps={step + 1}, "
                f"policy_loss={metrics['policy_loss']:.4f}"
            )

            # Save checkpoint
            self.agent.save(
                self.config.training.weights_dir,
                name=f"policy_ppo_episode{episode + 1}",
            )

        self.metrics.close()
        logger.info("PPO training complete.")

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        self.agent.load(Path(checkpoint_path).parent, Path(checkpoint_path).stem)
        logger.info(f"Resumed from {checkpoint_path}")
