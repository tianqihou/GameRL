"""
Tests for the hybrid and universal action spaces.

Covers:
- HybridActionSpace wrapper (standalone, legacy mode)
- ActionMapper with dynamic touch types (legacy mode)
- TransformerPolicy with continuous_dim > 0
- UniversalActionSpace (universal mode)
- TouchExecutor (universal mode)
- TransformerPolicy.for_universal factory
- PPOAgent hybrid action selection
"""

import numpy as np
import pytest
import torch

from gamerl.utils.actions import (
    ActionSpace,
    HybridActionSpace,
    UniversalActionSpace,
    TouchType,
    HARDWARE_KEYS,
)
from gamerl.profiles.base import TouchAction, GameProfile
from gamerl.profiles.peacekeeper import PeacekeeperEliteProfile
from gamerl.profiles.honor_of_kings import HonorOfKingsProfile
from gamerl.environment.device import ActionMapper


# ── HybridActionSpace (standalone, legacy) ─────────────────────────


class TestHybridActionSpace:
    """Tests for the HybridActionSpace wrapper (legacy mode)."""

    def test_pure_discrete(self):
        discrete = ActionSpace(
            movements=["up", "down"],
            actions=["attack", "idle"],
        )
        hybrid = HybridActionSpace(discrete, [])
        assert hybrid.is_hybrid() is False
        assert hybrid.continuous_dim == 0
        assert hybrid.vocab_size == 4

    def test_with_continuous(self):
        discrete = ActionSpace(
            movements=["up", "down"],
            actions=["attack", "idle"],
        )
        hybrid = HybridActionSpace(discrete, ["look_dx", "look_dy"])
        assert hybrid.is_hybrid() is True
        assert hybrid.continuous_dim == 2
        assert hybrid.vocab_size == 4

    def test_encode_decode(self):
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
        hybrid = HybridActionSpace(ActionSpace(), ["dx", "dy"])
        params = np.array([2.0, -3.0], dtype=np.float32)
        clamped = hybrid.clamp_continuous(params)
        assert clamped[0] == 1.0
        assert clamped[1] == -1.0

        params_t = torch.tensor([2.0, -3.0])
        clamped_t = hybrid.clamp_continuous(params_t)
        assert clamped_t[0].item() == 1.0
        assert clamped_t[1].item() == -1.0

    def test_sample_continuous(self):
        hybrid = HybridActionSpace(ActionSpace(), ["dx", "dy"])
        params = hybrid.sample_continuous()
        assert params.shape == (2,)
        assert np.all(params >= -1.0)
        assert np.all(params <= 1.0)

    def test_params_to_dict_and_back(self):
        hybrid = HybridActionSpace(ActionSpace(), ["dx", "dy"])
        params = np.array([0.5, -0.3], dtype=np.float32)
        d = hybrid.params_to_dict(params)
        assert d["dx"] == pytest.approx(0.5)
        assert d["dy"] == pytest.approx(-0.3)
        params_back = hybrid.dict_to_params(d)
        np.testing.assert_array_almost_equal(params, params_back)


# ── ActionMapper dynamic types (legacy) ────────────────────────────


class TestActionMapperDynamic:
    """Test ActionMapper with 'look' and 'dynamic_joystick' types (legacy)."""

    def test_look_with_params(self):
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))
        cmd = mapper.get_action_command("aim", {"look_dx": 1.0, "look_dy": 0.0})
        assert "d 0 960 540" in cmd
        assert "m 0 1360 540" in cmd

    def test_look_with_negative_params(self):
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
        assert "760" in cmd
        assert "340" in cmd

    def test_look_zero_params_degrades_to_tap(self):
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
        assert "d 0 960 540" in cmd
        assert "m " not in cmd

    def test_look_clamps_params(self):
        mapping = {
            "aim": TouchAction(
                type="look",
                coords=(960, 540, 400),
                duration_ms=50,
                param_keys=("look_dx", "look_dy"),
            ),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))
        cmd = mapper.get_action_command("aim", {"look_dx": 5.0, "look_dy": 0.0})
        assert "1360" in cmd

    def test_dynamic_joystick(self):
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
        assert "d 1 300 850" in cmd
        assert "m 1 375 775" in cmd

    def test_static_actions_ignore_params(self):
        mapping = {
            "shoot": TouchAction(type="tap", coords=(100, 200), duration_ms=50),
        }
        mapper = ActionMapper(mapping, resolution=(1920, 1080))
        cmd = mapper.get_action_command("shoot", {"look_dx": 1.0, "look_dy": 1.0})
        assert "d 0 100 200" in cmd
        assert "m " not in cmd


# ── UniversalActionSpace ───────────────────────────────────────────


class TestUniversalActionSpace:
    """Test the universal action space (game-agnostic)."""

    def test_discrete_size(self):
        assert UniversalActionSpace.DISCRETE_SIZE == 7
        assert UniversalActionSpace.vocab_size == 7

    def test_continuous_dim(self):
        assert UniversalActionSpace.CONTINUOUS_DIM == 5
        assert UniversalActionSpace.CONTINUOUS_PARAMS == ["x", "y", "dx", "dy", "duration"]

    def test_bos_token(self):
        assert UniversalActionSpace.BOS_TOKEN == TouchType.WAIT.value

    def test_decode_params_center(self):
        """Params [0,0,0,0,*] decode to screen center."""
        params = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        px, py, pdx, pdy, dur = UniversalActionSpace.decode_params(params, (1080, 2160))
        assert px == 540   # center x
        assert py == 1080  # center y
        assert pdx == 0
        assert pdy == 0

    def test_decode_params_corners(self):
        """Params [-1,-1,...] and [1,1,...] decode to corners."""
        res = (1080, 2160)
        params_tl = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        px, py, _, _, _ = UniversalActionSpace.decode_params(params_tl, res)
        assert px == 0
        assert py == 0

        params_br = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        px, py, _, _, _ = UniversalActionSpace.decode_params(params_br, res)
        assert px == 1079  # clamped to w-1
        assert py == 2159  # clamped to h-1

    def test_decode_params_duration(self):
        """Duration maps [-1,1] to [50,2000] ms."""
        params = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        _, _, _, _, dur = UniversalActionSpace.decode_params(params, (1080, 2160))
        assert dur == 50

        params = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        _, _, _, _, dur = UniversalActionSpace.decode_params(params, (1080, 2160))
        assert dur == 2000

    def test_decode_params_swipe_delta(self):
        """dx/dy scale by min(w,h)/2."""
        res = (1080, 2160)
        half_min = 540  # min(1080,2160)/2
        params = np.array([0.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)
        _, _, pdx, pdy, _ = UniversalActionSpace.decode_params(params, res)
        assert pdx == 540
        assert pdy == -540

    def test_sample(self):
        touch_type, params = UniversalActionSpace.sample()
        assert 0 <= touch_type < 7
        assert params.shape == (5,)
        assert np.all(params >= -1.0)
        assert np.all(params <= 1.0)

    def test_neutral_params(self):
        params = UniversalActionSpace.neutral_params()
        assert params.shape == (5,)
        assert params[0] == 0.0  # center x
        assert params[1] == 0.0  # center y

    def test_clamp_continuous(self):
        params = np.array([2.0, -3.0, 0.5, 0.0, 10.0], dtype=np.float32)
        clamped = UniversalActionSpace.clamp_continuous(params)
        assert clamped[0] == 1.0
        assert clamped[1] == -1.0
        assert clamped[4] == 1.0

    def test_describe(self):
        params = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        desc = UniversalActionSpace.describe(0, params, (1080, 2160))
        assert "TAP" in desc
        assert "540" in desc

    def test_key_index_mapping(self):
        assert UniversalActionSpace.decode_key_index(-1.0) == 0
        idx = UniversalActionSpace.decode_key_index(1.0)
        assert idx == len(HARDWARE_KEYS) - 1

    def test_touch_type_enum(self):
        assert TouchType.TAP.value == 0
        assert TouchType.LONG_PRESS.value == 1
        assert TouchType.SWIPE.value == 2
        assert TouchType.DRAG.value == 3
        assert TouchType.DOUBLE_TAP.value == 4
        assert TouchType.KEY_EVENT.value == 5
        assert TouchType.WAIT.value == 6


# ── TransformerPolicy universal mode ───────────────────────────────


class TestTransformerPolicyUniversal:
    """Test TransformerPolicy in universal mode."""

    def test_for_universal_factory(self):
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy.for_universal(
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
        )
        assert policy.vocab_size == 7
        assert policy.continuous_dim == 5
        assert policy.continuous_head is not None
        assert policy.continuous_log_std is not None

    def test_from_profile_universal(self):
        from gamerl.models.transformer import TransformerPolicy

        profile = HonorOfKingsProfile()
        policy = TransformerPolicy.from_profile(profile, feature_dim=64, d_model=128, n_layers=2, n_heads=4)
        assert policy.vocab_size == 7
        assert policy.continuous_dim == 5

    def test_forward_universal(self):
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=128, n_layers=2, n_heads=4,
        )
        policy.eval()

        batch_size, seq_len = 2, 5
        img_feats = torch.randn(batch_size, seq_len, 64)
        actions = torch.randint(0, 7, (batch_size, seq_len))

        logits, values, cont_mean = policy(img_feats, actions)

        assert logits.shape == (batch_size, seq_len, 7)
        assert values.shape == (batch_size, seq_len, 1)
        assert cont_mean is not None
        assert cont_mean.shape == (batch_size, seq_len, 5)
        assert cont_mean.min() >= -1.0
        assert cont_mean.max() <= 1.0

    def test_get_last_step_universal(self):
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=128, n_layers=2, n_heads=4,
        )
        policy.eval()

        img_feats = torch.randn(2, 5, 64)
        actions = torch.randint(0, 7, (2, 5))

        logits_last, value_last, cont_last = policy.get_last_step(img_feats, actions)

        assert logits_last.shape == (2, 7)
        assert value_last.shape == (2, 1)
        assert cont_last is not None
        assert cont_last.shape == (2, 5)

    def test_universal_gradients(self):
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=128, n_layers=2, n_heads=4,
        )
        img_feats = torch.randn(1, 3, 64)
        actions = torch.randint(0, 7, (1, 3))

        logits, values, cont_mean = policy(img_feats, actions)
        loss = logits.sum() + values.sum() + cont_mean.sum()
        loss.backward()

        assert policy.continuous_head.weight.grad is not None
        assert policy.continuous_log_std.requires_grad


# ── PPO Agent universal mode ───────────────────────────────────────


class TestPPOUniversal:
    """Test PPO agent with universal action space."""

    def test_select_action_universal(self):
        from gamerl.config import AgentConfig
        from gamerl.models.transformer import TransformerPolicy
        from gamerl.agent.ppo import PPOAgent

        config = AgentConfig()
        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=128, n_layers=2, n_heads=4,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PPOAgent(config, policy, backbone=None, device=device)

        image_features = np.random.randn(5, 64).astype(np.float32)
        action_history = np.array([6, 0, 2, 4, 1], dtype=np.int64)  # universal tokens

        action, log_prob, value, cont_params, cont_log_prob = agent.select_action(
            image_features, action_history
        )

        assert 0 <= action < 7
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert cont_params is not None
        assert cont_params.shape == (5,)
        assert np.all(cont_params >= -1.0)
        assert np.all(cont_params <= 1.0)

    def test_store_and_update_universal(self):
        from gamerl.config import AgentConfig
        from gamerl.models.transformer import TransformerPolicy
        from gamerl.agent.ppo import PPOAgent

        config = AgentConfig()
        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=128, n_layers=2, n_heads=4,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PPOAgent(config, policy, backbone=None, device=device)

        for i in range(8):
            agent.store_transition(
                image_features=np.random.randn(64).astype(np.float32),
                action=i % 7,
                log_prob=-2.0,
                value=0.5,
                reward=1.0,
                done=(i == 7),
                continuous_params=np.array([0.3, -0.4, 0.1, 0.0, -0.5], dtype=np.float32),
                continuous_log_prob=-1.0,
            )

        metrics = agent.update(last_value=0.0)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert np.isfinite(metrics["total_loss"])

    def test_from_profile_universal(self):
        from gamerl.config import AgentConfig
        from gamerl.agent.ppo import PPOAgent

        profile = HonorOfKingsProfile()
        config = AgentConfig()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PPOAgent.from_profile(
            profile=profile,
            config=config,
            backbone=None,
            device=device,
            feature_dim=64,
            d_model=128,
            n_layers=2,
            n_heads=4,
        )

        assert agent.policy.vocab_size == 7
        assert agent.policy.continuous_dim == 5
