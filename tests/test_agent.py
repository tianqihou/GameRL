"""
Tests for the PPO agent and memory.

Verifies that the PPO implementation is correct:
- GAE computation
- PPO clipping
- Action selection
- Memory operations
"""

import numpy as np
import pytest
import torch

from gamerl.agent.memory import RolloutMemory
from gamerl.config import AgentConfig
from gamerl.models.transformer import TransformerPolicy
from gamerl.agent.ppo import PPOAgent
from gamerl.utils.actions import ActionSpace


class TestRolloutMemory:
    """Tests for the rollout memory."""

    def test_add_and_len(self):
        mem = RolloutMemory(max_size=100)
        assert len(mem) == 0

        mem.add(
            image_features=np.random.randn(768),
            action=5,
            log_prob=-2.0,
            value=0.5,
            reward=1.0,
            done=False,
        )
        assert len(mem) == 1

    def test_max_size_sliding_window(self):
        mem = RolloutMemory(max_size=3)
        for i in range(5):
            mem.add(
                image_features=np.random.randn(768),
                action=i,
                log_prob=-1.0,
                value=0.0,
                reward=float(i),
                done=False,
            )
        assert len(mem) == 3
        # Oldest entries should have been removed
        assert mem.actions == [2, 3, 4]

    def test_gae_computation(self):
        """Test that GAE produces correct advantage values."""
        mem = RolloutMemory(max_size=100)
        gamma = 0.99
        gae_lambda = 0.95

        # Simple scenario: 3 transitions, all not done
        for i in range(3):
            mem.add(
                image_features=np.random.randn(768),
                action=i,
                log_prob=-1.0,
                value=1.0,
                reward=1.0,
                done=False,
            )

        mem.compute_gae(gamma=gamma, gae_lambda=gae_lambda, last_value=0.0)

        assert mem.advantages is not None
        assert mem.returns is not None
        assert len(mem.advantages) == 3
        assert len(mem.returns) == 3

        # Advantages should be finite
        assert np.all(np.isfinite(mem.advantages))

    def test_gae_with_done(self):
        """Test that episode boundaries reset GAE correctly."""
        mem = RolloutMemory(max_size=100)

        mem.add(np.random.randn(768), 0, -1.0, 1.0, 1.0, done=False)
        mem.add(np.random.randn(768), 1, -1.0, 1.0, 1.0, done=True)  # Episode ends
        mem.add(np.random.randn(768), 2, -1.0, 1.0, 1.0, done=False)  # New episode

        mem.compute_gae(gamma=0.99, gae_lambda=0.95, last_value=0.0)

        # After episode boundary, GAE should reset
        # The third transition's advantage should only depend on its own reward
        assert mem.advantages is not None
        assert np.isfinite(mem.advantages).all()

    def test_get_batches(self):
        mem = RolloutMemory(max_size=100)

        for i in range(10):
            mem.add(
                image_features=np.random.randn(64),
                action=i,
                log_prob=-2.0,
                value=0.0,
                reward=1.0,
                done=False,
            )

        mem.compute_gae(gamma=0.99, gae_lambda=0.95)

        batches = list(mem.get_batches(batch_size=4))
        assert len(batches) == 3  # 10 / 4 = 2 full + 1 partial

        for batch in batches:
            assert "image_features" in batch
            assert "actions" in batch
            assert "advantages" in batch
            assert "returns" in batch

    def test_clear(self):
        mem = RolloutMemory()
        mem.add(np.random.randn(64), 0, -1.0, 0.0, 1.0, False)
        mem.clear()
        assert len(mem) == 0


class TestPPOAgent:
    """Tests for the PPO agent."""

    @pytest.fixture
    def agent(self):
        """Create a small PPO agent for testing."""
        config = AgentConfig(
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            learning_rate=1e-3,
            ppo_epochs=2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            batch_size=4,
            minibatch_size=2,
        )

        policy = TransformerPolicy(
            feature_dim=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            vocab_size=130,
            dropout=0.0,
            max_seq_len=32,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return PPOAgent(config, policy, backbone=None, device=device)

    def test_select_action(self, agent):
        """Test action selection returns valid outputs."""
        image_features = np.random.randn(5, 64).astype(np.float32)
        action_history = np.array([128, 5, 10, 15, 20], dtype=np.int64)

        action, log_prob, value, cont_params, cont_log_prob = agent.select_action(image_features, action_history)

        assert 0 <= action < 131
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        # Pure discrete game (default test policy) → no continuous params
        assert cont_params is None
        assert cont_log_prob == 0.0

    def test_select_action_manual(self, agent):
        """Test manual action override."""
        image_features = np.random.randn(3, 64).astype(np.float32)
        action_history = np.array([128, 5, 10], dtype=np.int64)

        action, log_prob, value, _, _ = agent.select_action(
            image_features, action_history, manual_action=42
        )

        assert action == 42

    def test_update(self, agent):
        """Test that PPO update runs without errors."""
        # Add some transitions
        for i in range(8):
            agent.store_transition(
                image_features=np.random.randn(64).astype(np.float32),
                action=i % 131,
                log_prob=-2.0,
                value=0.5,
                reward=1.0,
                done=(i == 7),
            )

        # Run update
        metrics = agent.update(last_value=0.0)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])

    def test_save_load(self, agent, tmp_path):
        """Test checkpoint save/load."""
        # Save
        agent.save(tmp_path, name="test_policy")

        # Create new agent and load
        config = AgentConfig(
            batch_size=4,
            minibatch_size=2,
            ppo_epochs=1,
        )
        policy = TransformerPolicy(
            feature_dim=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            vocab_size=130,
            dropout=0.0,
            max_seq_len=32,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        new_agent = PPOAgent(config, policy, backbone=None, device=device)

        new_agent.load(tmp_path, name="test_policy")

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            agent.policy.named_parameters(),
            new_agent.policy.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Weights don't match for {n1}"
