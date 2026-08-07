"""
Tests for the hybrid action space (discrete + continuous).

Covers:
- HybridActionSpace wrapper
- ActionMapper with dynamic touch types ("look", "dynamic_joystick")
- TransformerPolicy with continuous_dim > 0
- GameProfile.continuous_params property
- PPOAgent hybrid action selection
"""

import numpy as np
import pytest
import torch

from gamerl.utils.actions import ActionSpace, HybridActionSpace
from gamerl.profiles.base import TouchAction, GameProfile
from gamerl.profiles.peacekeeper import PeacekeeperEliteProfile
from gamerl.profiles.genshin import GenshinImpactProfile
from gamerl.profiles.mini_world import MiniWorldProfile
from gamerl.profiles.honor_of_kings import HonorOfKingsProfile
from gamerl.profiles.roco_kingdom import RocoKingdomProfile
from gamerl.environment.device import ActionMapper


# ── HybridActionSpace ──────────────────────────────────────────────


class TestHybridActionSpace:
    """Tests for the HybridActionSpace wrapper."""

    def test_pure_discrete(self):
        """HybridActionSpace with empty continuous params behaves like ActionSpace."""
        discrete = ActionSpace(
            movements=["up", "down"],
            actions=["attack", "idle"],
        )
        hybrid = HybridActionSpace(discrete, [])

        assert hybrid.is_hybrid() is False
        assert hybrid.continuous_dim == 0
        assert hybrid.vocab_size == 4

    def test_with_continuous(self):
        """HybridActionSpace with continuous params."""
        discrete = ActionSpace(
            movements=["up", "down"],
            actions=["attack", "idle"],
        )
        hybrid = HybridActionSpace(discrete, ["look_dx", "look_dy"])

        assert hybrid.is_hybrid() is True
        assert hybrid.continuous_dim == 2
        assert hybrid.vocab_size == 4  # discrete size unchanged

    def test_encode_decode(self):
        """Encode/decode delegates to discrete space."""
        discrete = ActionSpace(
            movements=["up", "down"],
            actions=["attack", "idle"],
        )
        hybrid = HybridActionSpace(discrete, ["look_dx"])

        token = hybrid.encode("up", "attack")
        assert token == 0

        m, a = hybrid.decode(token)
        assert m == "up"
        assert a == "attack"

    def test_clamp_continuous(self):
        """Clamp continuous params to [-1, 1]."""
        hybrid = HybridActionSpace(ActionSpace(), ["dx", "dy"])

        # Numpy
        params = np.array([2.0, -3.0], dtype=np.float32)
        clamped = hybrid.clamp_continuous(params)
        assert clamped[0] == 1.0
        assert clamped[1] == -1.0

        # Torch
        params_t = torch.tensor([2.0, -3.0])
        clamped_t = hybrid.clamp_continuous(params_t)
        assert clamped_t[0].item() == 1.0
        assert clamped_t[1].item() == -1.0

    def test_sample_continuous(self):
        """Sample continuous params within [-1, 1]."""
        hybrid = HybridActionSpace(ActionSpace(), ["dx", "dy"])
        params = hybrid.sample_continuous()

        assert params.shape == (2,)
        assert np.all(params >= -1.0)
        assert np.all(params <= 1.0)

    def test_sample_continuous_empty(self):
        """Empty continuous space returns empty array."""
        hybrid = HybridActionSpace(ActionSpace(), [])
        params = hybrid.sample_continuous()
        assert params.shape == (0,)

    def test_params_to_dict_and_back(self):
        """Convert between param vector and dict."""
        hybrid = HybridActionSpace(ActionSpace(), ["dx", "dy"])
        params = np.array([0.5, -0.3], dtype=np.float32)

        d = hybrid.params_to_dict(params)
        assert d["dx"] == pytest.approx(0.5)
        assert d["dy"] == pytest.approx(-0.3)

        params_back = hybrid.dict_to_params(d)
        np.testing.assert_array_almost_equal(params, params_back)

    def test_from_profile_pure_discrete(self):
        """from_profile for a pure-discrete game."""
        profile = HonorOfKingsProfile()
        hybrid = HybridActionSpace.from_profile(profile)

        assert hybrid.is_hybrid() is False
        assert hybrid.continuous_dim == 0

    def test_from_profile_hybrid(self):
        """from_profile for a hybrid game."""
        profile = PeacekeeperEliteProfile()
        hybrid = HybridActionSpace.from_profile(profile)

        assert hybrid.is_hybrid() is True
        assert hybrid.continuous_dim == 2
        assert "look_dx" in hybrid.continuous_param_names
        assert "look_dy" in hybrid.continuous_param_names


# ── Profile continuous_params ──────────────────────────────────────


class TestProfileContinuousParams:
    """Test that each profile correctly declares continuous params."""

    def test_honor_of_kings_pure_discrete(self):
        profile = HonorOfKingsProfile()
        assert profile.continuous_params == []
        assert profile.is_hybrid is False
        assert profile.num_continuous_params == 0

    def test_peacekeeper_hybrid(self):
        profile = PeacekeeperEliteProfile()
        assert "look_dx" in profile.continuous_params
        assert "look_dy" in profile.continuous_params
        assert profile.is_hybrid is True
        assert profile.num_continuous_params == 2

    def test_genshin_hybrid(self):
        profile = GenshinImpactProfile()
        assert "aim_dx" in profile.continuous_params
        assert "aim_dy" in profile.continuous_params
        assert profile.is_hybrid is True

    def test_mini_world_hybrid(self):
        profile = MiniWorldProfile()
        assert "look_dx" in profile.continuous_params
        assert "look_dy" in profile.continuous_params
        assert profile.is_hybrid is True

    def test_roco_kingdom_pure_discrete(self):
        profile = RocoKingdomProfile()
        assert profile.continuous_params == []
        assert profile.is_hybrid is False

    def test_to_dict_includes_continuous(self):
        profile = PeacekeeperEliteProfile()
        d = profile.to_dict()
        assert "continuous_params" in d
        assert "is_hybrid" in d
        assert d["is_hybrid"] is True


# ── ActionMapper dynamic types ─────────────────────────────────────


class TestActionMapperDynamic:
    """Test ActionMapper with "look" and "dynamic_joystick" types."""

    def test_look_with_params(self):
        """"look" action computes swipe from center + params."""
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        # dx=1, dy=0 → swipe right
        cmd = mapper.get_action_command("aim", {"look_dx": 1.0, "look_dy": 0.0})
        assert "d 0 960 540" in cmd
        assert "m 0 1360 540" in cmd  # 960 + 1.0 * 400 = 1360

    def test_look_with_negative_params(self):
        """"look" with negative params swipes in opposite direction."""
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        cmd = mapper.get_action_command("aim", {"look_dx": -0.5, "look_dy": -0.5})
        # end_x = 960 + (-0.5) * 400 = 760
        # end_y = 540 + (-0.5) * 400 = 340
        assert "760" in cmd
        assert "340" in cmd

    def test_look_zero_params_degrades_to_tap(self):
        """"look" with zero params does a tap at center."""
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        cmd = mapper.get_action_command("aim", {"look_dx": 0.0, "look_dy": 0.0})
        # Should be a tap (down + up, no move)
        assert "d 0 960 540" in cmd
        assert "m " not in cmd  # no move command

    def test_look_clamps_params(self):
        """Params beyond [-1, 1] are clamped."""
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        # dx=5.0 should be clamped to 1.0
        cmd = mapper.get_action_command("aim", {"look_dx": 5.0, "look_dy": 0.0})
        assert "1360" in cmd  # 960 + 1.0 * 400

    def test_look_no_params(self):
        """"look" without continuous_params dict does a center tap."""
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        cmd = mapper.get_action_command("aim")
        assert "d 0 960 540" in cmd

    def test_dynamic_joystick(self):
        """"dynamic_joystick" uses pointer id 1."""
        mapping = {
            "move": TouchAction(
                type="dynamic_joystick",
                coords=(300, 850, 150),
                duration_ms=100,
                param_keys=("move_dx", "move_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        cmd = mapper.get_action_command("move", {"move_dx": 0.5, "move_dy": -0.5})
        # Should use pointer id 1 (joystick)
        assert "d 1 300 850" in cmd
        # end_x = 300 + 0.5 * 150 = 375
        # end_y = 850 + (-0.5) * 150 = 775
        assert "m 1 375 775" in cmd

    def test_static_actions_ignore_params(self):
        """Static action types (tap) ignore continuous_params."""
        mapping = {
            "shoot": TouchAction(type="tap", coords=(100, 200), duration_ms=50),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))

        cmd = mapper.get_action_command("shoot", {"look_dx": 1.0, "look_dy": 1.0})
        assert "d 0 100 200" in cmd
        assert "m " not in cmd

    def test_from_profile_dynamic(self):
        """ActionMapper.from_profile includes dynamic actions."""
        profile = PeacekeeperEliteProfile()
        mapper = ActionMapper.from_profile(profile)

        # "aim" should be a "look" type
        cmd = mapper.get_action_command("aim", {"look_dx": 1.0, "look_dy": 0.0})
        assert cmd != ""
        assert "m " in cmd  # has a move command (dynamic)


# ── TransformerPolicy with continuous_dim ──────────────────────────


class TestTransformerPolicyContinuous:
    """Test TransformerPolicy with continuous action head."""

    def test_continuous_head_exists(self):
        """Policy with continuous_dim > 0 has continuous head."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=80,
            continuous_dim=2,
        )

        assert policy.continuous_head is not None
        assert policy.continuous_log_std is not None
        assert policy.continuous_dim == 2

    def test_pure_discrete_no_continuous_head(self):
        """Policy with continuous_dim=0 has no continuous head."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=130,
            continuous_dim=0,
        )

        assert policy.continuous_head is None
        assert policy.continuous_log_std is None

    def test_forward_with_continuous(self):
        """Forward pass outputs continuous mean when continuous_dim > 0."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=80,
            continuous_dim=2,
        )
        policy.eval()

        batch_size, seq_len = 2, 5
        img_feats = torch.randn(batch_size, seq_len, 64)
        actions = torch.randint(0, 80, (batch_size, seq_len))

        logits, values, cont_mean = policy(img_feats, actions)

        assert logits.shape == (batch_size, seq_len, 80)
        assert values.shape == (batch_size, seq_len, 1)
        assert cont_mean is not None
        assert cont_mean.shape == (batch_size, seq_len, 2)
        # Tanh squashes to [-1, 1]
        assert cont_mean.min() >= -1.0
        assert cont_mean.max() <= 1.0

    def test_get_last_step_with_continuous(self):
        """get_last_step returns continuous mean for last step."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=80,
            continuous_dim=2,
        )
        policy.eval()

        batch_size, seq_len = 2, 5
        img_feats = torch.randn(batch_size, seq_len, 64)
        actions = torch.randint(0, 80, (batch_size, seq_len))

        logits_last, value_last, cont_last = policy.get_last_step(img_feats, actions)

        assert logits_last.shape == (batch_size, 80)
        assert value_last.shape == (batch_size, 1)
        assert cont_last is not None
        assert cont_last.shape == (batch_size, 2)

    def test_continuous_gradients(self):
        """Gradients flow through continuous head."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=80,
            continuous_dim=2,
        )

        img_feats = torch.randn(1, 3, 64)
        actions = torch.randint(0, 80, (1, 3))

        logits, values, cont_mean = policy(img_feats, actions)
        loss = logits.sum() + values.sum() + cont_mean.sum()
        loss.backward()

        # continuous_head participates in forward → gets gradients
        assert policy.continuous_head.weight.grad is not None
        # continuous_log_std is a learned param used only in PPO's Normal
        # distribution loss, not in forward() — so no grad from forward pass.
        # We just verify it's a proper learnable parameter.
        assert policy.continuous_log_std.requires_grad


# ── PPO Agent hybrid action selection ──────────────────────────────


class TestPPOHybrid:
    """Test PPO agent with hybrid action space."""

    def test_select_action_hybrid(self):
        """PPO agent with continuous_dim > 0 returns continuous params."""
        from gamerl.config import AgentConfig
        from gamerl.models.transformer import TransformerPolicy
        from gamerl.agent.ppo import PPOAgent

        config = AgentConfig()
        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=80,
            continuous_dim=2,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PPOAgent(config, policy, backbone=None, device=device)

        image_features = np.random.randn(5, 64).astype(np.float32)
        action_history = np.array([0, 5, 10, 15, 20], dtype=np.int64)

        action, log_prob, value, cont_params, cont_log_prob = agent.select_action(
            image_features, action_history
        )

        assert 0 <= action < 80
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        # Hybrid → continuous params should be present
        assert cont_params is not None
        assert cont_params.shape == (2,)
        assert np.all(cont_params >= -1.0)
        assert np.all(cont_params <= 1.0)
        # Continuous log prob should be a real number
        assert isinstance(cont_log_prob, float)

    def test_store_and_update_hybrid(self):
        """PPO update works with hybrid transitions."""
        from gamerl.config import AgentConfig
        from gamerl.models.transformer import TransformerPolicy
        from gamerl.agent.ppo import PPOAgent

        config = AgentConfig()
        policy = TransformerPolicy(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=80,
            continuous_dim=2,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PPOAgent(config, policy, backbone=None, device=device)

        # Store transitions with continuous params
        for i in range(8):
            agent.store_transition(
                image_features=np.random.randn(64).astype(np.float32),
                action=i % 80,
                log_prob=-2.0,
                value=0.5,
                reward=1.0,
                done=(i == 7),
                continuous_params=np.array([0.3, -0.4], dtype=np.float32),
                continuous_log_prob=-0.5,
            )

        metrics = agent.update(last_value=0.0)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert np.isfinite(metrics["total_loss"])
