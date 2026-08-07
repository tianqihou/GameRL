"""
Tests for neural network models.

Verifies that the modernized models produce correct output shapes
and can do forward/backward passes.
"""

import pytest
import torch

from gamerl.config import ModelConfig
from gamerl.models.transformer import TransformerPolicy, RotaryPositionalEmbedding
from gamerl.models.state_judgment import StateJudgmentModel
from gamerl.utils.actions import ActionSpace, BOS_TOKEN
from gamerl.utils.masks import create_causal_mask, create_masks


class TestActionSpace:
    """Tests for the action space."""

    def test_vocab_size(self):
        space = ActionSpace()
        assert space.vocab_size == 130  # 10 movements * 13 actions

    def test_encode_decode_roundtrip(self):
        space = ActionSpace()
        from gamerl.utils.actions import Movement, ActionType

        for m in Movement:
            for a in ActionType:
                token = space.encode(m.value, a.value)
                m2, a2 = space.decode(token)
                assert m.value == m2
                assert a.value == a2

    def test_decode_str(self):
        space = ActionSpace()
        from gamerl.utils.actions import Movement, ActionType

        token = space.encode(Movement.UP.value, ActionType.ATTACK.value)
        assert space.decode_to_str(token) == "上移_攻击"

    def test_bos_token(self):
        assert BOS_TOKEN == 128


class TestMasks:
    """Tests for attention masks."""

    def test_causal_mask_shape(self):
        mask = create_causal_mask(10, torch.device("cpu"))
        assert mask.shape == (1, 10, 10)

    def test_causal_mask_values(self):
        mask = create_causal_mask(4, torch.device("cpu"))
        # Position 0 can only attend to position 0
        assert mask[0, 0, 0] == True
        assert mask[0, 0, 1] == False
        # Position 1 can attend to 0 and 1
        assert mask[0, 1, 0] == True
        assert mask[0, 1, 1] == True
        assert mask[0, 1, 2] == False

    def test_create_masks(self):
        src = torch.tensor([[1, 2, 3, -1]])  # Last is padding
        trg = torch.tensor([[1, 2, 3, -1]])
        src_mask, trg_mask = create_masks(src, trg, torch.device("cpu"))

        assert src_mask.shape == (1, 1, 4)
        assert trg_mask.shape == (1, 4, 4)
        # Padding position should be masked
        assert src_mask[0, 0, 3] == False  # Position 3 is padding


class TestRotaryPositionalEmbedding:
    """Tests for RoPE."""

    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(d_model=64, max_seq_len=128)
        x = torch.randn(2, 10, 64)
        out = rope(x, seq_len=10)
        assert out.shape == (2, 10, 64)


class TestTransformerPolicy:
    """Tests for the policy network."""

    @pytest.fixture
    def small_policy(self):
        """Create a small policy for testing."""
        return TransformerPolicy(
            feature_dim=36 * 768,  # 6*6 * 768
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=131,
            dropout=0.0,
            max_seq_len=64,
        )

    def test_forward_output_shapes(self, small_policy):
        batch_size, seq_len = 2, 10
        image_features = torch.randn(batch_size, seq_len, 36 * 768)
        actions = torch.randint(0, 130, (batch_size, seq_len))
        attn_mask = create_causal_mask(seq_len, torch.device("cpu")).squeeze(0)

        logits, values, cont_mean = small_policy(image_features, actions, attn_mask=attn_mask)

        assert logits.shape == (batch_size, seq_len, 131)
        assert values.shape == (batch_size, seq_len, 1)
        # Pure discrete (continuous_dim=0) → no continuous output
        assert cont_mean is None

    def test_get_last_step(self, small_policy):
        batch_size, seq_len = 2, 10
        image_features = torch.randn(batch_size, seq_len, 36 * 768)
        actions = torch.randint(0, 130, (batch_size, seq_len))
        attn_mask = create_causal_mask(seq_len, torch.device("cpu")).squeeze(0)

        logits_last, value_last, cont_last = small_policy.get_last_step(
            image_features, actions, attn_mask=attn_mask
        )

        assert logits_last.shape == (batch_size, 131)
        assert value_last.shape == (batch_size, 1)
        assert cont_last is None

    def test_backward_pass(self, small_policy):
        """Test that gradients flow correctly."""
        batch_size, seq_len = 2, 5
        image_features = torch.randn(batch_size, seq_len, 36 * 768)
        actions = torch.randint(0, 130, (batch_size, seq_len))
        attn_mask = create_causal_mask(seq_len, torch.device("cpu")).squeeze(0)

        logits, values, _ = small_policy(image_features, actions, attn_mask=attn_mask)

        loss = logits.sum() + values.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in small_policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self, small_policy):
        """Verify the model has a reasonable number of parameters."""
        n_params = small_policy.count_parameters()
        assert n_params > 0


class TestStateJudgmentModel:
    """Tests for the state judgment model."""

    def test_classify_single_frame(self):
        model = StateJudgmentModel(
            feature_dim=36 * 768,
            d_model=128,
            n_layers=2,
            n_heads=4,
            num_classes=6,
        )

        features = torch.randn(4, 36 * 768)
        logits = model.classify_single_frame(features)
        assert logits.shape == (4, 6)
