"""
Integration tests for the universal action space data pipeline.

Covers the end-to-end chain that was fixed after the universal action
space refactor:
- DataCollector: universal/legacy mode recording
- DataPreprocessor: continuous params + profile BOS
- GameSequenceDataset: continuous param loading and collation
- RolloutMemory: continuous param serialization
- export_onnx: vocab_size parameterization
- StateJudgmentModel: universal vocab default
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch

from gamerl.agent.memory import RolloutMemory
from gamerl.data.dataset import GameSequenceDataset, collate_sequences
from gamerl.data.preprocessor import DataPreprocessor
from gamerl.models.state_judgment import StateJudgmentModel
from gamerl.profiles import get_profile
from gamerl.utils.actions import (
    BOS_TOKEN,
    TouchType,
    UniversalActionSpace,
)


# ---------------------------------------------------------------------------
# RolloutMemory continuous param serialization
# ---------------------------------------------------------------------------

class TestRolloutMemoryContinuousSerialization:
    """RolloutMemory save/load must preserve continuous params."""

    def test_save_load_with_continuous_params(self, tmp_path):
        mem = RolloutMemory(max_size=100)
        for i in range(5):
            mem.add(
                image_features=np.random.randn(64).astype(np.float32),
                action=i % 7,
                log_prob=-1.5,
                value=0.5,
                reward=1.0,
                done=(i == 4),
                continuous_params=np.random.uniform(-1, 1, 5).astype(np.float32),
                continuous_log_prob=-0.8,
            )

        path = str(tmp_path / "rollout.npz")
        mem.save(path)

        mem2 = RolloutMemory(max_size=100)
        mem2.load(path)

        assert len(mem2) == 5
        assert mem2.continuous_params[0] is not None
        # Object arrays from save/load need element-wise comparison
        np.testing.assert_allclose(
            np.asarray(mem2.continuous_params[0], dtype=np.float32),
            np.asarray(mem.continuous_params[0], dtype=np.float32),
            atol=1e-6,
        )
        assert mem2.continuous_log_probs[0] == pytest.approx(-0.8)

    def test_save_load_without_continuous_params(self, tmp_path):
        """Legacy (pure discrete) saves must still work."""
        mem = RolloutMemory(max_size=100)
        for i in range(3):
            mem.add(
                image_features=np.random.randn(32).astype(np.float32),
                action=i,
                log_prob=-2.0,
                value=0.0,
                reward=0.5,
                done=False,
            )

        path = str(tmp_path / "rollout.npz")
        mem.save(path)

        mem2 = RolloutMemory(max_size=100)
        mem2.load(path)

        assert len(mem2) == 3
        # Backward compat: no continuous data → filled with None / 0.0
        assert all(cp is None for cp in mem2.continuous_params)
        assert all(clp == 0.0 for clp in mem2.continuous_log_probs)


# ---------------------------------------------------------------------------
# DataPreprocessor universal mode
# ---------------------------------------------------------------------------

class TestDataPreprocessorUniversal:
    """DataPreprocessor handles universal action records correctly."""

    def _make_episode(self, tmp_path, universal=True, n_frames=5):
        """Create a fake episode directory with actions.jsonl + dummy images."""
        ep_dir = tmp_path / "episode_001"
        ep_dir.mkdir()

        records = []
        for i in range(n_frames):
            rec = {
                "frame": i,
                "image": f"{i}.jpg",
                "reward": 0.01,
                "done": i == n_frames - 1,
                "timestamp": 1000000.0 + i,
            }
            if universal:
                rec["touch_type"] = i % 7
                rec["continuous_params"] = [0.1 * i] * 5
            else:
                rec["action_token"] = i * 10
                rec["movement"] = "上移"
                rec["action"] = "攻击"
            records.append(rec)

        with open(ep_dir / "actions.jsonl", "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Create dummy images (small RGB)
        from PIL import Image
        for i in range(n_frames):
            img = Image.new("RGB", (8, 8), color=(i * 50, 0, 0))
            img.save(ep_dir / f"{i}.jpg")

        return ep_dir

    def test_universal_episode_produces_continuous_params(self, tmp_path):
        ep_dir = self._make_episode(tmp_path, universal=True)

        backbone = MagicMock()
        backbone_output = torch.randn(1, 36, 768)
        backbone.return_value = backbone_output

        preprocessor = DataPreprocessor(backbone, bos_token=UniversalActionSpace.BOS_TOKEN)
        features, actions, cont = preprocessor.process_episode(ep_dir)

        assert actions[0] == UniversalActionSpace.BOS_TOKEN  # BOS prepended
        assert len(actions) == 6  # BOS + 5 frames
        assert cont is not None
        assert cont.shape == (6, 5)  # BOS row + 5 frames
        # BOS row should be neutral params
        np.testing.assert_allclose(
            cont[0], UniversalActionSpace.neutral_params(), atol=1e-6
        )

    def test_legacy_episode_no_continuous(self, tmp_path):
        ep_dir = self._make_episode(tmp_path, universal=False)

        backbone = MagicMock()
        backbone.return_value = torch.randn(1, 36, 768)

        preprocessor = DataPreprocessor(backbone, bos_token=BOS_TOKEN)
        features, actions, cont = preprocessor.process_episode(ep_dir)

        assert actions[0] == BOS_TOKEN  # legacy BOS
        assert len(actions) == 6
        assert cont is None  # no continuous params in legacy data

    def test_from_profile_factory(self):
        profile = get_profile("honor_of_kings")
        backbone = MagicMock()
        preprocessor = DataPreprocessor.from_profile(backbone, profile)
        assert preprocessor.bos_token == UniversalActionSpace.BOS_TOKEN

    def test_process_and_save_includes_continuous(self, tmp_path):
        ep_dir = self._make_episode(tmp_path, universal=True)

        backbone = MagicMock()
        backbone.return_value = torch.randn(1, 36, 768)

        preprocessor = DataPreprocessor(backbone)
        out_path = preprocessor.process_and_save(ep_dir)

        data = np.load(out_path)
        assert "continuous_params" in data
        assert data["continuous_params"].shape == (6, 5)


# ---------------------------------------------------------------------------
# GameSequenceDataset with continuous params
# ---------------------------------------------------------------------------

class TestDatasetContinuousParams:
    """Dataset loads and chunks continuous params correctly."""

    def _make_npz(self, tmp_path, with_continuous=True, seq_len=20):
        """Create a fake preprocessed.npz."""
        feature_dim = 64
        image_features = np.random.randn(seq_len, feature_dim).astype(np.float32)
        action_sequence = np.arange(seq_len + 1, dtype=np.int64) % 7

        arrays = {
            "image_features": image_features,
            "action_sequence": action_sequence,
        }
        if with_continuous:
            arrays["continuous_params"] = np.random.randn(
                seq_len + 1, 5
            ).astype(np.float32)

        ep_dir = tmp_path / "ep1"
        ep_dir.mkdir()
        np.savez_compressed(ep_dir / "preprocessed.npz", **arrays)
        return ep_dir

    def test_dataset_loads_continuous(self, tmp_path):
        self._make_npz(tmp_path, with_continuous=True)
        ds = GameSequenceDataset(tmp_path, chunk_size=10, stride=10)

        assert ds.has_continuous
        item = ds[0]
        assert "continuous_params" in item
        assert "target_continuous_params" in item
        assert item["continuous_params"].shape[-1] == 5
        assert item["continuous_params"].shape[0] == item["actions"].shape[0]

    def test_dataset_without_continuous(self, tmp_path):
        self._make_npz(tmp_path, with_continuous=False)
        ds = GameSequenceDataset(tmp_path, chunk_size=10, stride=10)

        assert not ds.has_continuous
        item = ds[0]
        assert "continuous_params" not in item

    def test_collate_with_continuous(self, tmp_path):
        self._make_npz(tmp_path, with_continuous=True)
        ds = GameSequenceDataset(tmp_path, chunk_size=10, stride=10)

        items = [ds[i] for i in range(min(2, len(ds)))]
        batch = collate_sequences(items)

        assert "continuous_params" in batch
        assert "target_continuous_params" in batch
        assert batch["continuous_params"].shape[-1] == 5
        assert batch["padding_mask"].shape == batch["actions"].shape

    def test_collate_without_continuous(self, tmp_path):
        self._make_npz(tmp_path, with_continuous=False)
        ds = GameSequenceDataset(tmp_path, chunk_size=10, stride=10)

        items = [ds[i] for i in range(min(2, len(ds)))]
        batch = collate_sequences(items)

        assert "continuous_params" not in batch


# ---------------------------------------------------------------------------
# StateJudgmentModel universal vocab
# ---------------------------------------------------------------------------

class TestStateJudgmentUniversal:
    """StateJudgmentModel defaults are compatible with universal action space."""

    def test_default_vocab_is_7(self):
        model = StateJudgmentModel(feature_dim=64, d_model=32, n_layers=1, n_heads=2)
        assert model.action_embed.num_embeddings == 7

    def test_classify_single_frame_uses_bos(self):
        """Single-frame classification must not produce out-of-range tokens."""
        model = StateJudgmentModel(
            feature_dim=64, d_model=32, n_layers=1, n_heads=2, num_classes=4,
        )
        features = torch.randn(2, 64)
        logits = model.classify_single_frame(features)
        assert logits.shape == (2, 4)

    def test_custom_vocab_size_override(self):
        """Explicit vocab_size (e.g. legacy 130) still works."""
        model = StateJudgmentModel(
            feature_dim=64, d_model=32, n_layers=1, n_heads=2, vocab_size=130,
        )
        assert model.action_embed.num_embeddings == 130


# ---------------------------------------------------------------------------
# ONNX export vocab_size parameterization
# ---------------------------------------------------------------------------

class TestExportOnnxVocab:
    """export_to_onnx infers vocab_size from the model."""

    def test_vocab_inferred_from_model(self):
        """Dummy actions must be within the model's vocab range."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=32, n_layers=1, n_heads=2, max_seq_len=16,
        )

        # The policy's embedding has 7 rows — exporting with randint(0, 130) would crash
        assert policy.action_embed.num_embeddings == 7

        # Verify the vocab_size inference logic works
        from gamerl.inference.export_onnx import export_to_onnx
        # We can't actually export without onnx installed, but we can verify
        # the parameter is accepted
        import inspect
        sig = inspect.signature(export_to_onnx)
        assert "vocab_size" in sig.parameters


# ---------------------------------------------------------------------------
# PolicyTrainer construction
# ---------------------------------------------------------------------------

class TestPolicyTrainerConstruction:
    """PolicyTrainer builds components from profile (not hardcoded vocab)."""

    def test_trainer_uses_profile(self):
        """Trainer must resolve profile and build universal policy."""
        from gamerl.config import Config

        config = Config()
        config.model.backbone = "convnext_tiny"
        config.model.pretrained = False
        config.model.d_model = 32
        config.model.n_layers = 1
        config.model.n_heads = 2
        config.training.log_dir = os.devnull  # suppress TensorBoard

        from gamerl.training.trainer import PolicyTrainer
        trainer = PolicyTrainer(config, device="cpu")

        # Profile resolved
        assert trainer.profile is not None
        assert trainer.profile.is_universal

        # Policy configured for universal action space
        assert trainer.policy.vocab_size == UniversalActionSpace.DISCRETE_SIZE
        assert trainer.policy.continuous_dim == UniversalActionSpace.CONTINUOUS_DIM

    def test_trainer_peacekeeper_profile(self):
        """Different game profile also gets universal policy."""
        from gamerl.config import Config

        config = Config()
        config.game.name = "peacekeeper"
        config.model.pretrained = False
        config.model.d_model = 32
        config.model.n_layers = 1
        config.model.n_heads = 2
        config.training.log_dir = os.devnull

        from gamerl.training.trainer import PolicyTrainer
        trainer = PolicyTrainer(config, device="cpu")

        assert "和平精英" in trainer.profile.display_name
        assert trainer.policy.vocab_size == 7


# ---------------------------------------------------------------------------
# Structured state fusion (vision pipeline → policy input)
# ---------------------------------------------------------------------------

class TestStructuredStateFusion:
    """StructuredState must flow from vision through to the policy."""

    def test_policy_with_state_dim(self):
        """Policy with state_dim>0 creates state_proj and fuses state."""
        from gamerl.models.transformer import TransformerPolicy

        state_dim = 65  # default StructuredState.vector_dim()
        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=32, n_layers=1, n_heads=2,
            max_seq_len=16, state_dim=state_dim,
        )
        assert policy.state_proj is not None
        assert policy.state_dim == state_dim

        # Forward with structured state
        img = torch.randn(2, 8, 64)
        acts = torch.randint(0, 7, (2, 8))
        struct = torch.randn(2, 8, state_dim)
        logits, values, cont = policy(img, acts, structured_state=struct)
        assert logits.shape == (2, 8, 7)
        assert cont.shape == (2, 8, 5)

    def test_policy_without_state_dim_ignores_state(self):
        """Policy with state_dim=0 silently ignores structured_state."""
        from gamerl.models.transformer import TransformerPolicy

        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=32, n_layers=1, n_heads=2, max_seq_len=16,
        )
        assert policy.state_proj is None

        img = torch.randn(2, 8, 64)
        acts = torch.randint(0, 7, (2, 8))
        struct = torch.randn(2, 8, 65)
        # Must not crash even if structured_state is passed
        logits, values, cont = policy(img, acts, structured_state=struct)
        assert logits.shape == (2, 8, 7)

    def test_select_action_with_structured_state(self):
        """PPOAgent.select_action accepts and forwards structured_state."""
        from gamerl.agent.ppo import PPOAgent
        from gamerl.config import AgentConfig
        from gamerl.models.transformer import TransformerPolicy

        state_dim = 65
        policy = TransformerPolicy.for_universal(
            feature_dim=64, d_model=32, n_layers=1, n_heads=2,
            max_seq_len=16, state_dim=state_dim,
        )
        agent = PPOAgent(AgentConfig(batch_size=4), policy, backbone=None, device="cpu")

        img = np.random.randn(5, 64).astype(np.float32)
        hist = np.array([6, 0, 1, 2, 3], dtype=np.int64)
        struct = np.random.randn(5, state_dim).astype(np.float32)

        action, log_prob, value, cont, clp = agent.select_action(
            img, hist, structured_state=struct,
        )
        assert 0 <= action < 7
        assert cont is not None and cont.shape == (5,)

    def test_memory_structured_state_roundtrip(self, tmp_path):
        """RolloutMemory stores, batches, and serializes structured states."""
        mem = RolloutMemory(max_size=100)
        state_dim = 65
        for i in range(6):
            mem.add(
                image_features=np.random.randn(64).astype(np.float32),
                action=i % 7,
                log_prob=-1.0,
                value=0.5,
                reward=1.0,
                done=(i == 5),
                structured_state=np.random.randn(state_dim).astype(np.float32),
            )

        mem.compute_gae(gamma=0.99, gae_lambda=0.95)
        batches = list(mem.get_batches(batch_size=4))
        assert "structured_states" in batches[0]
        assert batches[0]["structured_states"].shape[-1] == state_dim

        # Save/load roundtrip
        path = str(tmp_path / "rollout.npz")
        mem.save(path)
        mem2 = RolloutMemory()
        mem2.load(path)
        assert mem2.structured_states[0] is not None
        assert len(mem2.structured_states) == 6

    def test_memory_without_structured_state(self, tmp_path):
        """Memory without structured states must not include the key."""
        mem = RolloutMemory(max_size=100)
        for i in range(4):
            mem.add(np.random.randn(32), i, -1.0, 0.0, 1.0, False)

        mem.compute_gae(gamma=0.99, gae_lambda=0.95)
        batches = list(mem.get_batches(batch_size=4))
        assert "structured_states" not in batches[0]

    def test_trainer_computes_state_dim(self):
        """PolicyTrainer computes state_dim from vision config."""
        from gamerl.config import Config

        config = Config()
        config.model.pretrained = False
        config.model.d_model = 32
        config.model.n_layers = 1
        config.model.n_heads = 2
        config.training.log_dir = os.devnull
        # vision.enabled defaults to True → state_dim should be > 0
        config.vision.enabled = True

        from gamerl.training.trainer import PolicyTrainer
        trainer = PolicyTrainer(config, device="cpu")

        assert trainer.state_dim > 0
        assert trainer.policy.state_dim == trainer.state_dim
        assert trainer.policy.state_proj is not None

    def test_trainer_vision_disabled_no_state(self):
        """With vision.enabled=False, policy has no state fusion."""
        from gamerl.config import Config

        config = Config()
        config.model.pretrained = False
        config.model.d_model = 32
        config.model.n_layers = 1
        config.model.n_heads = 2
        config.training.log_dir = os.devnull
        config.vision.enabled = False

        from gamerl.training.trainer import PolicyTrainer
        trainer = PolicyTrainer(config, device="cpu")

        assert trainer.state_dim == 0
        assert trainer.policy.state_proj is None


# ---------------------------------------------------------------------------
# Config: imitation & strategic_rewards sections
# ---------------------------------------------------------------------------

class TestConfigSections:
    """imitation and strategic_rewards YAML sections must be live."""

    def test_imitation_config_loaded(self, tmp_path):
        import yaml
        from gamerl.config import Config

        cfg = {
            "imitation": {"enabled": True, "bc_epochs": 42,
                          "dataset_path": "/tmp/demo"},
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")

        config = Config.from_yaml(path)
        assert config.imitation.enabled is True
        assert config.imitation.bc_epochs == 42
        assert config.imitation.dataset_path == "/tmp/demo"

    def test_strategic_rewards_loaded(self, tmp_path):
        import yaml
        from gamerl.config import Config

        cfg = {
            "strategic_rewards": {"objective_progress": 0.5,
                                   "survival": 0.02, "exploration": 0.005},
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")

        config = Config.from_yaml(path)
        assert config.strategic_rewards.objective_progress == 0.5
        assert config.strategic_rewards.survival == 0.02

    def test_reward_clip_from_config(self, tmp_path):
        import yaml
        from gamerl.config import Config

        cfg = {"rewards": {"clip_min": -5.0, "clip_max": 5.0,
                            "kill_hero": 8.0}}
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")

        config = Config.from_yaml(path)
        assert config.rewards.clip_min == -5.0
        assert config.rewards.clip_max == 5.0
        assert config.rewards.events["kill_hero"] == 8.0


# ---------------------------------------------------------------------------
# RewardShaper: strategic rewards consumption
# ---------------------------------------------------------------------------

class TestStrategicRewards:
    """RewardShaper must apply strategic reward components."""

    def _make_shaper(self, strategic=None):
        from gamerl.environment.reward import RewardShaper
        return RewardShaper(
            reward_events={"kill": 5.0, "normal": 0.01, "other": -0.003},
            terminal_events=["death"],
            strategic_rewards=strategic,
        )

    def test_survival_bonus(self):
        from gamerl.config import StrategicRewardsConfig
        sr = StrategicRewardsConfig(survival=0.02)
        shaper = self._make_shaper(strategic=sr)

        reward, done, _ = shaper.compute_reward(action=0)
        # default event = "normal" (0.01) + survival (0.02)
        assert reward == pytest.approx(0.03, abs=1e-6)

    def test_objective_progress_bonus(self):
        from gamerl.config import StrategicRewardsConfig
        sr = StrategicRewardsConfig(objective_progress=0.5)
        shaper = self._make_shaper(strategic=sr)

        # Force a positive event via callback
        shaper.event_callback = lambda p, c, a: "kill"
        reward, _, _ = shaper.compute_reward(action=0)
        # kill (5.0) + objective_progress (0.5)
        assert reward == pytest.approx(5.5, abs=1e-6)

    def test_no_strategic_rewards_unchanged(self):
        """Without strategic config, behavior is identical to before."""
        shaper = self._make_shaper(strategic=None)
        reward, _, _ = shaper.compute_reward(action=0)
        assert reward == pytest.approx(0.01, abs=1e-6)

    def test_reward_clipping(self):
        """Rewards must be clipped to the configured bounds."""
        from gamerl.environment.reward import RewardShaper
        shaper = RewardShaper(
            reward_events={"jackpot": 999.0, "normal": 0.01},
            clip_min=-5.0, clip_max=5.0,
        )
        shaper.event_callback = lambda p, c, a: "jackpot"
        reward, _, _ = shaper.compute_reward(action=0)
        assert reward == 5.0
