"""
Tests for game profiles.

Verifies that each game profile correctly defines its state classes,
reward events, screen regions, and uses the universal action space.
Also tests the profile registry and factory function.
"""

import pytest

from gamerl.profiles import (
    GameProfile,
    HonorOfKingsProfile,
    PeacekeeperEliteProfile,
    GenshinImpactProfile,
    MiniWorldProfile,
    RocoKingdomProfile,
    get_profile,
    list_profiles,
    PROFILES,
    ALIASES,
)
from gamerl.profiles.base import ScreenRegion
from gamerl.utils.actions import UniversalActionSpace


# ── Universal action space (shared across all profiles) ─────────────


class TestUniversalActionMode:
    """All profiles use the universal action space."""

    @pytest.mark.parametrize("profile_cls", list(PROFILES.values()))
    def test_universal_mode(self, profile_cls):
        profile = profile_cls()
        assert profile.action_mode == "universal"
        assert profile.is_universal is True

    @pytest.mark.parametrize("profile_cls", list(PROFILES.values()))
    def test_vocab_size_is_7(self, profile_cls):
        """Universal action space has exactly 7 touch types."""
        profile = profile_cls()
        assert profile.vocab_size == UniversalActionSpace.DISCRETE_SIZE
        assert profile.vocab_size == 7

    @pytest.mark.parametrize("profile_cls", list(PROFILES.values()))
    def test_continuous_params_universal(self, profile_cls):
        """All profiles use the same 5 universal continuous params."""
        profile = profile_cls()
        assert profile.continuous_params == UniversalActionSpace.CONTINUOUS_PARAMS
        assert profile.num_continuous_params == 5
        assert profile.is_hybrid is True

    @pytest.mark.parametrize("profile_cls", list(PROFILES.values()))
    def test_bos_token_is_wait(self, profile_cls):
        """BOS token in universal mode is WAIT (6)."""
        profile = profile_cls()
        assert profile.bos_token == UniversalActionSpace.BOS_TOKEN

    @pytest.mark.parametrize("profile_cls", list(PROFILES.values()))
    def test_to_dict_includes_action_mode(self, profile_cls):
        profile = profile_cls()
        d = profile.to_dict()
        assert d["action_mode"] == "universal"
        assert d["vocab_size"] == 7
        assert "continuous_params" in d
        assert len(d["continuous_params"]) == 5


# ── Honor of Kings ──────────────────────────────────────────────────


class TestHonorOfKingsProfile:

    @pytest.fixture
    def profile(self):
        return HonorOfKingsProfile()

    def test_display_name(self, profile):
        assert "王者荣耀" in profile.display_name

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 8
        assert "normal" in profile.state_classes
        assert "kill_hero" in profile.state_classes

    def test_resolution(self, profile):
        w, h = profile.resolution
        assert w > 0 and h > 0

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "normal" in events
        assert any(v > 0 for v in events.values())
        assert any(v < 0 for v in events.values())

    def test_terminal_events(self, profile):
        for te in profile.terminal_events:
            assert te in profile.reward_events

    def test_state_classes_subset_of_reward_events(self, profile):
        for sc in profile.state_classes:
            assert sc in profile.reward_events

    def test_screen_regions(self, profile):
        regions = profile.screen_regions
        assert "hp_bar" in regions
        assert "minimap" in regions
        assert "skills" in regions


# ── Peacekeeper Elite ───────────────────────────────────────────────


class TestPeacekeeperEliteProfile:

    @pytest.fixture
    def profile(self):
        return PeacekeeperEliteProfile()

    def test_display_name(self, profile):
        assert "和平精英" in profile.display_name

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 5
        assert "combat" in profile.state_classes
        assert "parachuting" in profile.state_classes

    def test_screen_regions(self, profile):
        assert "crosshair" in profile.screen_regions
        assert "minimap" in profile.screen_regions

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "kill_enemy" in events
        assert "won_match" in events
        assert events["won_match"] > events["kill_enemy"]

    def test_terminal_events(self, profile):
        assert "got_killed" in profile.terminal_events
        assert "won_match" in profile.terminal_events


# ── Genshin Impact ──────────────────────────────────────────────────


class TestGenshinImpactProfile:

    @pytest.fixture
    def profile(self):
        return GenshinImpactProfile()

    def test_display_name(self, profile):
        assert "原神" in profile.display_name

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 4
        assert "overworld" in profile.state_classes
        assert "combat" in profile.state_classes

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "defeat_boss" in events
        assert "party_wiped" in events
        assert events["party_wiped"] < 0

    def test_terminal_events(self, profile):
        assert "party_wiped" in profile.terminal_events


# ── Mini World ──────────────────────────────────────────────────────


class TestMiniWorldProfile:

    @pytest.fixture
    def profile(self):
        return MiniWorldProfile()

    def test_display_name(self, profile):
        assert "迷你世界" in profile.display_name

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 7
        assert "survival_day" in profile.state_classes
        assert "creative" in profile.state_classes

    def test_detection_classes(self, profile):
        classes = profile.detection_classes
        assert "player" in classes
        assert "mob" in classes
        assert "boss" in classes

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "collect_resource" in events
        assert "defeat_boss" in events
        assert events["defeat_boss"] > events["defeat_mob"]

    def test_no_terminal_events(self, profile):
        assert profile.terminal_events == []

    def test_state_classes_subset_of_reward_events(self, profile):
        for sc in profile.state_classes:
            assert sc in profile.reward_events


# ── Roco Kingdom ────────────────────────────────────────────────────


class TestRocoKingdomProfile:

    @pytest.fixture
    def profile(self):
        return RocoKingdomProfile()

    def test_display_name(self, profile):
        assert "洛克王国" in profile.display_name

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 5
        assert "exploration" in profile.state_classes
        assert "battle" in profile.state_classes
        assert "catching" in profile.state_classes

    def test_detection_classes(self, profile):
        classes = profile.detection_classes
        assert "pet_wild" in classes
        assert "pet_enemy" in classes
        assert "chest" in classes

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "catch_pet" in events
        assert "win_battle" in events
        assert events["battle_lost"] < 0

    def test_terminal_events(self, profile):
        assert "battle_lost" in profile.terminal_events

    def test_state_classes_subset_of_reward_events(self, profile):
        for sc in profile.state_classes:
            assert sc in profile.reward_events


# ── Registry ────────────────────────────────────────────────────────


class TestProfileRegistry:

    def test_list_profiles(self):
        profiles = list_profiles()
        assert "honor_of_kings" in profiles
        assert "peacekeeper" in profiles
        assert "genshin" in profiles
        assert "mini_world" in profiles
        assert "roco_kingdom" in profiles

    def test_get_profile_by_name(self):
        profile = get_profile("honor_of_kings")
        assert isinstance(profile, HonorOfKingsProfile)

    def test_get_profile_by_alias(self):
        profile = get_profile("wzry")
        assert isinstance(profile, HonorOfKingsProfile)

        profile2 = get_profile("王者荣耀")
        assert isinstance(profile2, HonorOfKingsProfile)

    def test_get_profile_unknown(self):
        with pytest.raises(ValueError, match="Unknown game profile"):
            get_profile("nonexistent_game")

    def test_all_profiles_in_registry(self):
        assert len(PROFILES) >= 5

    def test_all_aliases_resolve(self):
        for alias, target in ALIASES.items():
            profile = get_profile(alias)
            assert isinstance(profile, PROFILES[target])

    def test_all_profiles_use_universal_mode(self):
        """Every registered profile should use universal action mode."""
        for name, cls in PROFILES.items():
            profile = cls()
            assert profile.action_mode == "universal", (
                f"{name} should use universal action mode"
            )
            assert profile.vocab_size == 7
            assert profile.num_continuous_params == 5
