"""
Tests for the RewardShaper and game-specific reward system.

Verifies that:
1. RewardShaper correctly maps events to rewards
2. Terminal events trigger done=True
3. Strategic directive weights are applied
4. Profile-based factory works correctly
5. Each game has distinct, sensible reward structures
"""

import numpy as np
import pytest

from gamerl.environment.reward import RewardShaper
from gamerl.profiles import (
    HonorOfKingsProfile,
    PeacekeeperEliteProfile,
    GenshinImpactProfile,
    MiniWorldProfile,
    RocoKingdomProfile,
)


class TestRewardShaper:
    """Tests for RewardShaper core functionality."""

    @pytest.fixture
    def shaper(self):
        return RewardShaper(
            reward_events={
                "kill_hero": 5.0,
                "killed": -2.0,
                "normal": 0.01,
                "other": -0.003,
            },
            state_classes=["kill_hero", "killed", "normal"],
            terminal_events=["killed"],
        )

    def test_normal_event(self, shaper):
        """Default event should be 'normal' with its reward."""
        reward, done, event = shaper.compute_reward()
        assert event == "normal"
        assert reward == pytest.approx(0.01)
        assert done is False

    def test_terminal_event(self, shaper):
        """Terminal events should set done=True."""
        # Use callback to force a terminal event
        shaper.event_callback = lambda prev, curr, action: "killed"
        reward, done, event = shaper.compute_reward()
        assert event == "killed"
        assert reward == pytest.approx(-2.0)
        assert done is True

    def test_non_terminal_negative_event(self, shaper):
        """Non-terminal negative event should give negative reward but done=False."""
        shaper.event_callback = lambda prev, curr, action: "kill_hero"
        reward, done, event = shaper.compute_reward()
        assert event == "kill_hero"
        assert reward == pytest.approx(5.0)
        assert done is False

    def test_strategic_weights_exact_match(self, shaper):
        """Strategic weights matching event name should be applied."""
        shaper.event_callback = lambda prev, curr, action: "kill_hero"
        reward, done, event = shaper.compute_reward(
            strategic_weights={"kill_hero": 3.0}
        )
        assert reward == pytest.approx(5.0 + 3.0)

    def test_strategic_weights_substring_match(self, shaper):
        """Strategic weights matching as substring should be applied."""
        shaper.event_callback = lambda prev, curr, action: "kill_hero"
        reward, done, event = shaper.compute_reward(
            strategic_weights={"kill": 1.0}
        )
        assert reward == pytest.approx(5.0 + 1.0)

    def test_strategic_weights_no_match(self, shaper):
        """Strategic weights that don't match should have no effect."""
        shaper.event_callback = lambda prev, curr, action: "normal"
        reward, done, event = shaper.compute_reward(
            strategic_weights={"kill_hero": 3.0}
        )
        assert reward == pytest.approx(0.01)

    def test_unknown_event_falls_back_to_other(self, shaper):
        """Events not in reward_events should use 'other' as fallback."""
        shaper.event_callback = lambda prev, curr, action: "unknown_event"
        reward, done, event = shaper.compute_reward()
        assert event == "unknown_event"
        assert reward == pytest.approx(-0.003)  # 'other' value

    def test_stats_tracking(self, shaper):
        """Stats should track event counts and rewards."""
        shaper.event_callback = lambda prev, curr, action: "normal"
        shaper.compute_reward()
        shaper.compute_reward()
        shaper.event_callback = lambda prev, curr, action: "kill_hero"
        shaper.compute_reward()

        stats = shaper.get_stats()
        assert stats["step_count"] == 3
        assert stats["event_counts"]["normal"] == 2
        assert stats["event_counts"]["kill_hero"] == 1
        assert stats["total_reward"] == pytest.approx(0.01 + 0.01 + 5.0)

    def test_reset_stats(self, shaper):
        """Reset should clear all stats."""
        shaper.compute_reward()
        shaper.reset_stats()
        stats = shaper.get_stats()
        assert stats["step_count"] == 0
        assert stats["total_reward"] == 0.0


class TestRewardShaperFromProfile:
    """Tests for creating RewardShaper from GameProfile."""

    def test_from_hok_profile(self):
        profile = HonorOfKingsProfile()
        shaper = RewardShaper.from_profile(profile)

        assert "kill_hero" in shaper.reward_events
        assert "kill_minion" in shaper.reward_events
        assert "kill_tower" in shaper.reward_events
        assert shaper.reward_events["kill_hero"] == 5.0
        assert "death" in shaper.terminal_events

    def test_from_peacekeeper_profile(self):
        profile = PeacekeeperEliteProfile()
        shaper = RewardShaper.from_profile(profile)

        assert "kill_enemy" in shaper.reward_events
        assert "won_match" in shaper.reward_events
        assert shaper.reward_events["kill_enemy"] == 10.0
        assert "got_killed" in shaper.terminal_events
        assert "won_match" in shaper.terminal_events

    def test_from_genshin_profile(self):
        profile = GenshinImpactProfile()
        shaper = RewardShaper.from_profile(profile)

        assert "defeat_boss" in shaper.reward_events
        assert "chest_opened" in shaper.reward_events
        assert shaper.reward_events["chest_opened"] == 2.0
        assert "party_wiped" in shaper.terminal_events

    def test_state_classes_match_profile(self):
        """Shaper's state_classes should match the profile's."""
        for profile_cls in [HonorOfKingsProfile, PeacekeeperEliteProfile, GenshinImpactProfile,
                            MiniWorldProfile, RocoKingdomProfile]:
            profile = profile_cls()
            shaper = RewardShaper.from_profile(profile)
            assert shaper.state_classes == profile.state_classes


class TestGameSpecificRewards:
    """Verify each game has distinct, non-overlapping reward structures."""

    def test_hok_has_moba_specific_events(self):
        profile = HonorOfKingsProfile()
        events = profile.reward_events
        # MOBA-specific events
        assert "kill_minion" in events
        assert "kill_tower" in events
        assert "kill_hero" in events
        assert "assist_kill" in events
        assert "attacked_by_tower" in events
        # Should NOT have FPS or RPG events
        assert "kill_enemy" not in events
        assert "chest_opened" not in events

    def test_peacekeeper_has_fps_specific_events(self):
        profile = PeacekeeperEliteProfile()
        events = profile.reward_events
        # FPS/Battle Royale specific events
        assert "kill_enemy" in events
        assert "down_enemy" in events
        assert "got_killed" in events
        assert "loot_item" in events
        assert "won_match" in events
        assert "reached_final_circle" in events
        # Should NOT have MOBA or RPG events
        assert "kill_minion" not in events
        assert "chest_opened" not in events

    def test_genshin_has_rpg_specific_events(self):
        profile = GenshinImpactProfile()
        events = profile.reward_events
        # RPG specific events
        assert "defeat_enemy" in events
        assert "defeat_boss" in events
        assert "chest_opened" in events
        assert "quest_completed" in events
        assert "material_collected" in events
        assert "exploration" in events
        assert "party_wiped" in events
        # Should NOT have MOBA or FPS events
        assert "kill_minion" not in events
        assert "kill_enemy" not in events

    def test_mini_world_has_sandbox_specific_events(self):
        profile = MiniWorldProfile()
        events = profile.reward_events
        # Sandbox specific events
        assert "collect_resource" in events
        assert "craft_item" in events
        assert "place_block" in events
        assert "defeat_mob" in events
        assert "defeat_boss" in events
        assert "survive_night" in events
        assert "tame_pet" in events
        assert "upgrade_tool" in events
        assert "harvest_crop" in events
        # Should NOT have MOBA, FPS, or RPG events
        assert "kill_minion" not in events
        assert "kill_enemy" not in events
        assert "chest_opened" not in events
        # Sandbox: no terminal events
        assert profile.terminal_events == []

    def test_roco_kingdom_has_pet_rpg_specific_events(self):
        profile = RocoKingdomProfile()
        events = profile.reward_events
        # Pet collection RPG specific events
        assert "catch_pet" in events
        assert "discover_new_pet" in events
        assert "pet_evolve" in events
        assert "win_battle" in events
        assert "defeat_world_boss" in events
        assert "pet_downed" in events
        assert "battle_lost" in events
        assert "complete_quest" in events
        assert "unlock_area" in events
        # Should NOT have MOBA, FPS, or sandbox events
        assert "kill_minion" not in events
        assert "kill_enemy" not in events
        assert "craft_item" not in events
        assert "place_block" not in events
        # Battle lost is terminal
        assert "battle_lost" in profile.terminal_events

    def test_all_games_have_normal_and_other(self):
        """Every game must define 'normal' and 'other' baseline events."""
        for profile_cls in [HonorOfKingsProfile, PeacekeeperEliteProfile, GenshinImpactProfile,
                            MiniWorldProfile, RocoKingdomProfile]:
            events = profile_cls().reward_events
            assert "normal" in events, f"{profile_cls.__name__} missing 'normal'"
            assert "other" in events, f"{profile_cls.__name__} missing 'other'"

    def test_reward_values_are_sensible(self):
        """Rewards should make sense: positive for good, negative for bad."""
        for profile_cls in [HonorOfKingsProfile, PeacekeeperEliteProfile, GenshinImpactProfile,
                            MiniWorldProfile, RocoKingdomProfile]:
            events = profile_cls().reward_events
            # Positive events (kills, wins, completions)
            positive_keys = [k for k, v in events.items() if v > 0 and k not in ("normal",)]
            assert len(positive_keys) >= 2, f"{profile_cls.__name__} should have positive rewards"
            # Negative events (deaths, failures)
            negative_keys = [k for k, v in events.items() if v < 0 and k != "other"]
            assert len(negative_keys) >= 1, f"{profile_cls.__name__} should have negative rewards"

    def test_no_reward_sharing_between_games(self):
        """No game-specific event should appear in all five games."""
        hok = set(HonorOfKingsProfile().reward_events.keys())
        pk = set(PeacekeeperEliteProfile().reward_events.keys())
        gs = set(GenshinImpactProfile().reward_events.keys())
        mw = set(MiniWorldProfile().reward_events.keys())
        rk = set(RocoKingdomProfile().reward_events.keys())

        # Only baseline events should be shared across all games
        shared = hok & pk & gs & mw & rk
        assert shared <= {"normal", "other"}, (
            f"Unexpected shared events: {shared - {'normal', 'other'}}"
        )
