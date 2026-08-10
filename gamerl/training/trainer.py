"""
Main training pipeline for the PPO policy network.

Combines data loading, model training, and checkpointing.
Supports:
- Mixed precision training (AMP)
- Gradient clipping
- TensorBoard logging
- Checkpoint resume
- Supervised pretraining + PPO fine-tuning
- Universal action space (discrete touch type + continuous params)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..agent.ppo import PPOAgent
from ..config import Config
from ..data.dataset import GameSequenceDataset, collate_sequences
from ..models.backbone import BackboneExtractor
from ..models.state_judgment import StateJudgmentModel
from ..profiles import get_profile
from ..utils.actions import UniversalActionSpace
from ..utils.logging import MetricsLogger
from ..utils.masks import create_causal_mask

logger = logging.getLogger("gamerl.training")


class PolicyTrainer:
    """
    Training pipeline for the policy network.

    Supports two training modes:
    1. Supervised pretraining: Learn from human/AI demonstration data
       (discrete cross-entropy + continuous MSE when continuous params exist)
    2. PPO fine-tuning: Reinforcement learning from game rewards

    The action mode (universal / legacy) is auto-detected from the game
    profile selected in ``config.game.name``.

    Args:
        config: Top-level configuration.
        device: Torch device.
    """

    def __init__(self, config: Config, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Resolve game profile (determines action mode, vocab size, BOS, etc.)
        self.profile = get_profile(config.game.name)
        logger.info(
            f"Game profile: {self.profile.display_name} "
            f"(action_mode={self.profile.action_mode}, "
            f"vocab={self.profile.vocab_size}, "
            f"continuous_params={self.profile.num_continuous_params})"
        )

        # Build backbone (frozen, for feature extraction)
        self.backbone = BackboneExtractor(
            backbone_name=config.model.backbone,
            grid_size=config.model.backbone_grid_size,
            pretrained=config.model.pretrained,
            freeze=True,
            use_half=config.model.backbone_half,
        ).to(self.device)

        feature_dim = self.backbone.get_flat_dim()

        # Compute structured-state dimension when the vision pipeline is enabled
        self.state_dim = 0
        if config.vision.enabled:
            from ..vision.state_builder import StructuredState
            self.state_dim = StructuredState.vector_dim(
                max_enemies=config.vision.max_enemies,
                max_towers=config.vision.max_towers,
                max_minions=config.vision.max_minions,
                num_skills=self.profile.num_skills,
            )

        # Build PPO agent (policy auto-configured for the profile's action mode)
        self.agent = PPOAgent.from_profile(
            self.profile,
            config=config.agent,
            backbone=None,  # Backbone is used separately during data collection
            device=self.device,
            feature_dim=feature_dim,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            state_dim=self.state_dim,
        )
        self.policy = self.agent.policy

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

        # vocab_size: config override > profile default (7 for universal)
        vocab_size = self.config.state_model.vocab_size
        if vocab_size is None:
            vocab_size = self.profile.vocab_size

        self.state_model = StateJudgmentModel(
            feature_dim=feature_dim,
            d_model=self.config.state_model.d_model,
            n_layers=self.config.state_model.n_layers,
            n_heads=self.config.state_model.n_heads,
            num_classes=self.config.state_model.num_classes,
            vocab_size=vocab_size,
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
        data_dir: Optional[str] = None,
        epochs: Optional[int] = None,
    ) -> None:
        """
        Supervised pretraining from demonstration data (behavior cloning).

        Trains the policy to predict the next action given the current
        state history.  Uses cross-entropy for the discrete touch type
        and MSE for continuous params (when the data contains them and
        the policy has a continuous head).

        When ``data_dir`` / ``epochs`` are omitted, falls back to the
        ``imitation`` config section (``dataset_path`` / ``bc_epochs``).

        Args:
            data_dir: Directory containing preprocessed .npz files.
            epochs: Number of training epochs (overrides config).
        """
        if data_dir is None:
            data_dir = self.config.imitation.dataset_path
        if epochs is None:
            epochs = (
                self.config.imitation.bc_epochs
                if self.config.imitation.enabled
                else self.config.training.epochs
            )
        logger.info(
            f"BC pretraining: data={data_dir}, epochs={epochs} "
            f"(imitation.enabled={self.config.imitation.enabled})"
        )

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

        has_continuous_head = self.policy.continuous_head is not None

        # AMP scaler
        scaler = torch.amp.GradScaler("cuda",
            enabled=self.config.training.use_amp and self.device.type == "cuda"
        )

        optimizer = self.agent.optimizer

        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_discrete_loss = 0.0
            epoch_cont_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                image_features = batch["image_features"].to(self.device)
                actions = batch["actions"].to(self.device)
                targets = batch["target_actions"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                # Continuous targets (universal mode data only)
                cont_targets = batch.get("target_continuous_params")
                if cont_targets is not None:
                    cont_targets = cont_targets.to(self.device)

                batch_size, seq_len, _ = image_features.shape

                # Create causal mask
                attn_mask = create_causal_mask(seq_len, self.device).squeeze(0)  # (S, S)

                # Forward pass with AMP
                with torch.amp.autocast("cuda", scaler.is_enabled()):
                    logits, values, cont_mean = self.policy(
                        image_features, actions,
                        attn_mask=attn_mask,
                        key_padding_mask=padding_mask,
                    )

                    # Discrete cross-entropy loss (ignore padding positions)
                    discrete_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-1,
                    )
                    loss = discrete_loss

                    # Continuous param regression (MSE on non-padding positions)
                    cont_loss = None
                    if (
                        has_continuous_head
                        and cont_mean is not None
                        and cont_targets is not None
                    ):
                        # valid mask: True where NOT padding
                        valid = ~padding_mask  # (B, S)
                        if valid.any():
                            cont_loss = F.mse_loss(
                                cont_mean[valid],      # (N, continuous_dim)
                                cont_targets[valid],   # (N, continuous_dim)
                            )
                            loss = loss + cont_loss

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
                epoch_discrete_loss += discrete_loss.item()
                if cont_loss is not None:
                    epoch_cont_loss += cont_loss.item()
                n_batches += 1
                global_step += 1

                # Log metrics
                if global_step % 10 == 0:
                    self.metrics.log_scalar("supervised/loss", loss.item(), global_step)
                    self.metrics.log_scalar(
                        "supervised/discrete_loss", discrete_loss.item(), global_step
                    )
                    if cont_loss is not None:
                        self.metrics.log_scalar(
                            "supervised/continuous_loss", cont_loss.item(), global_step
                        )

            n = max(n_batches, 1)
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"loss={epoch_loss / n:.4f} "
                f"(discrete={epoch_discrete_loss / n:.4f}"
                + (f", continuous={epoch_cont_loss / n:.4f}" if epoch_cont_loss > 0 else "")
                + ")"
            )

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

        # Attach a RewardShaper if the env doesn't have one — without it
        # PPO would train on reward=0.0 forever.
        if getattr(env, "reward_shaper", None) is None:
            from ..environment.reward import RewardShaper
            env.reward_shaper = RewardShaper.from_profile(
                self.profile,
                state_model=self.state_model,
                clip_min=self.config.rewards.clip_min,
                clip_max=self.config.rewards.clip_max,
                strategic_rewards=self.config.strategic_rewards,
            )
            # Apply YAML reward overrides on top of profile defaults
            if self.config.rewards.events:
                env.reward_shaper.reward_events.update(self.config.rewards.events)
            logger.info(
                "Attached RewardShaper to env "
                f"(clip=[{self.config.rewards.clip_min}, {self.config.rewards.clip_max}], "
                f"state_model={'yes' if self.state_model else 'no'})"
            )

        global_step = 0

        for episode in range(episodes):
            logger.info(f"PPO Episode {episode + 1}/{episodes}")

            # Collect rollout
            state = env.reset()
            episode_reward = 0.0

            for step in range(steps_per_episode):
                # Select action (5-tuple), passing structured state when available
                action, log_prob, value, cont_params, cont_log_prob = \
                    self.agent.select_action(
                        state.image_features,
                        state.action_history,
                        structured_state=state.structured_state,
                    )

                # Convert continuous params to dict for env.step
                params_dict = None
                if cont_params is not None:
                    params_dict = {
                        name: float(cont_params[i])
                        for i, name in enumerate(self.profile.continuous_params)
                    }

                # Take step
                prev_features = state.image_features[-1]
                prev_structured = (
                    state.structured_state[-1]
                    if state.structured_state is not None
                    else None
                )
                state, reward, done, info = env.step(action, params_dict)
                episode_reward += reward

                # Store transition (with continuous params + structured state)
                self.agent.store_transition(
                    image_features=prev_features,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                    continuous_params=cont_params,
                    continuous_log_prob=cont_log_prob,
                    structured_state=prev_structured,
                )

                global_step += 1

                if done:
                    break

            # Compute last value for GAE
            with torch.no_grad():
                last_structured = None
                if state.structured_state is not None:
                    last_structured = torch.FloatTensor(
                        state.structured_state[-1:]
                    ).unsqueeze(0).to(self.device)
                _, last_value_t, _ = self.agent.policy(
                    torch.FloatTensor(state.image_features[-1:]).unsqueeze(0).to(self.device),
                    torch.LongTensor(state.action_history[-1:]).unsqueeze(0).to(self.device),
                    structured_state=last_structured,
                )
                last_value = last_value_t.item()

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
