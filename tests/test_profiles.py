"""
Tests for game profiles.

Verifies that each game profile correctly defines its action space,
state classes, touch mappings, and screen regions. Also tests the
profile registry and factory function.
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
from gamerl.profiles.base import TouchAction, ScreenRegion
from gamerl.utils.actions import ActionSpace


class TestHonorOfKingsProfile:
    """Tests for the Honor of Kings profile."""

    @pytest.fixture
    def profile(self):
        return HonorOfKingsProfile()

    def test_display_name(self, profile):
        assert "王者荣耀" in profile.display_name

    def test_action_space_size(self, profile):
        assert len(profile.movements) == 10
        assert len(profile.actions) == 13
        assert profile.vocab_size == 130

    def test_bos_token(self, profile):
        assert profile.bos_token == 128

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 8
        assert "normal" in profile.state_classes
        assert "kill_hero" in profile.state_classes
        assert "kill_minion" in profile.state_classes
        assert "kill_tower" in profile.state_classes

    def test_idle_movements(self, profile):
        assert profile.is_idle_movement("无移动")
        assert profile.is_idle_movement("移动停")
        assert not profile.is_idle_movement("上移")

    def test_idle_actions(self, profile):
        assert profile.is_idle_action("无动作")
        assert not profile.is_idle_action("攻击")

    def test_touch_mapping_completeness(self, profile):
        """Every non-idle movement and action should have a touch mapping."""
        for m in profile.movements:
            if not profile.is_idle_movement(m):
                assert profile.get_touch_action(m) is not None, f"Missing touch for: {m}"

        for a in profile.actions:
            if not profile.is_idle_action(a):
                assert profile.get_touch_action(a) is not None, f"Missing touch for: {a}"

    def test_touch_action_types(self, profile):
        """Movements should be joystick type, buttons should be tap type."""
        up = profile.get_touch_action("上移")
        assert up is not None
        assert up.type == "joystick"
        assert len(up.coords) == 4  # (start_x, start_y, end_x, end_y)

        attack = profile.get_touch_action("攻击")
        assert attack is not None
        assert attack.type == "tap"
        assert len(attack.coords) == 2  # (x, y)

    def test_screen_regions(self, profile):
        regions = profile.screen_regions
        assert "hp_bar" in regions
        assert "minimap" in regions
        assert "skills" in regions

        hp = regions["hp_bar"]
        assert isinstance(hp, ScreenRegion)
        assert hp.width > 0 and hp.height > 0
        assert len(hp.box) == 4

    def test_resolution(self, profile):
        w, h = profile.resolution
        assert w > 0 and h > 0

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "normal" in events
        assert "other" in events
        # At least one positive and one negative reward
        assert any(v > 0 for v in events.values())
        assert any(v < 0 for v in events.values())

    def test_terminal_events(self, profile):
        terminals = profile.terminal_events
        # All terminal events must exist in reward_events
        for te in terminals:
            assert te in profile.reward_events, f"Terminal event '{te}' not in reward_events"

    def test_state_classes_subset_of_reward_events(self, profile):
        """Every state class should have a corresponding reward entry."""
        for sc in profile.state_classes:
            assert sc in profile.reward_events, (
                f"State class '{sc}' not found in reward_events"
            )

    def test_action_space_from_profile(self, profile):
        space = profile.action_space
        assert isinstance(space, ActionSpace)
        assert space.vocab_size == 130
        assert space.bos_token == 128

    def test_to_dict(self, profile):
        d = profile.to_dict()
        assert "display_name" in d
        assert "movements" in d
        assert "vocab_size" in d
        assert d["vocab_size"] == 130


class TestPeacekeeperEliteProfile:
    """Tests for the Peacekeeper Elite profile."""

    @pytest.fixture
    def profile(self):
        return PeacekeeperEliteProfile()

    def test_display_name(self, profile):
        assert "和平精英" in profile.display_name

    def test_action_space_size(self, profile):
        assert len(profile.movements) == 8
        assert len(profile.actions) == 10
        assert profile.vocab_size == 80

    def test_bos_token(self, profile):
        assert profile.bos_token == 0

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 5
        assert "combat" in profile.state_classes
        assert "parachuting" in profile.state_classes

    def test_idle_actions(self, profile):
        assert profile.is_idle_action("noop")
        assert not profile.is_idle_action("shoot")

    def test_touch_mapping(self, profile):
        shoot = profile.get_touch_action("shoot")
        assert shoot is not None
        assert shoot.type == "tap"

        forward = profile.get_touch_action("forward")
        assert forward is not None
        assert forward.type == "joystick"

    def test_screen_regions(self, profile):
        assert "crosshair" in profile.screen_regions
        assert "minimap" in profile.screen_regions


class TestGenshinImpactProfile:
    """Tests for the Genshin Impact profile."""

    @pytest.fixture
    def profile(self):
        return GenshinImpactProfile()

    def test_display_name(self, profile):
        assert "原神" in profile.display_name

    def test_action_space_size(self, profile):
        assert len(profile.movements) == 8
        assert len(profile.actions) == 10  # +aim (dynamic look)
        assert profile.vocab_size == 80

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 4
        assert "overworld" in profile.state_classes
        assert "combat" in profile.state_classes

    def test_touch_mapping(self, profile):
        attack = profile.get_touch_action("normal_attack")
        assert attack is not None
        assert attack.type == "tap"

    def test_charged_attack_duration(self, profile):
        charged = profile.get_touch_action("charged_attack")
        assert charged is not None
        assert charged.duration_ms == 500  # Long press


class TestMiniWorldProfile:
    """Tests for the Mini World profile."""

    @pytest.fixture
    def profile(self):
        return MiniWorldProfile()

    def test_display_name(self, profile):
        assert "迷你世界" in profile.display_name

    def test_action_space_size(self, profile):
        assert len(profile.movements) == 8
        assert len(profile.actions) == 9  # +look (dynamic camera)
        assert profile.vocab_size == 72

    def test_bos_token(self, profile):
        assert profile.bos_token == 0

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 7
        assert "survival_day" in profile.state_classes
        assert "combat" in profile.state_classes
        assert "creative" in profile.state_classes

    def test_idle_actions(self, profile):
        assert profile.is_idle_action("noop")
        assert not profile.is_idle_action("break_block")

    def test_touch_mapping(self, profile):
        break_action = profile.get_touch_action("break_block")
        assert break_action is not None
        assert break_action.type == "tap"
        assert break_action.duration_ms == 500  # long press to break

        place = profile.get_touch_action("place_block")
        assert place is not None
        assert place.type == "tap"

        forward = profile.get_touch_action("forward")
        assert forward is not None
        assert forward.type == "joystick"

    def test_screen_regions(self, profile):
        assert "hp_bar" in profile.screen_regions
        assert "hunger_bar" in profile.screen_regions
        assert "hotbar" in profile.screen_regions
        assert "minimap" in profile.screen_regions

    def test_detection_classes(self, profile):
        classes = profile.detection_classes
        assert "player" in classes
        assert "mob" in classes
        assert "resource_node" in classes
        assert "boss" in classes

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "collect_resource" in events
        assert "craft_item" in events
        assert "defeat_boss" in events
        assert "survive_night" in events
        assert events["defeat_boss"] > events["defeat_mob"]
        assert events["take_damage"] < 0

    def test_no_terminal_events(self, profile):
        """Sandbox game: no hard terminal state."""
        assert profile.terminal_events == []

    def test_state_classes_subset_of_reward_events(self, profile):
        for sc in profile.state_classes:
            assert sc in profile.reward_events, (
                f"State class '{sc}' not found in reward_events"
            )


class TestRocoKingdomProfile:
    """Tests for the Roco Kingdom profile."""

    @pytest.fixture
    def profile(self):
        return RocoKingdomProfile()

    def test_display_name(self, profile):
        assert "洛克王国" in profile.display_name

    def test_action_space_size(self, profile):
        assert len(profile.movements) == 5
        assert len(profile.actions) == 10
        assert profile.vocab_size == 50

    def test_bos_token(self, profile):
        assert profile.bos_token == 0

    def test_state_classes(self, profile):
        assert len(profile.state_classes) == 5
        assert "exploration" in profile.state_classes
        assert "battle" in profile.state_classes
        assert "catching" in profile.state_classes

    def test_idle_movements(self, profile):
        assert profile.is_idle_movement("stop")
        assert not profile.is_idle_movement("up")

    def test_idle_actions(self, profile):
        assert profile.is_idle_action("noop")
        assert not profile.is_idle_action("skill_1")

    def test_touch_mapping(self, profile):
        skill1 = profile.get_touch_action("skill_1")
        assert skill1 is not None
        assert skill1.type == "tap"

        catch = profile.get_touch_action("catch_pet")
        assert catch is not None
        assert catch.type == "tap"

        up = profile.get_touch_action("up")
        assert up is not None
        assert up.type == "joystick"

    def test_screen_regions(self, profile):
        regions = profile.screen_regions
        assert "my_pet_hp" in regions
        assert "enemy_pet_hp" in regions
        assert "energy_bar" in regions
        assert "skill_panel" in regions
        assert "minimap" in regions

    def test_detection_classes(self, profile):
        classes = profile.detection_classes
        assert "pet_wild" in classes
        assert "pet_enemy" in classes
        assert "chest" in classes

    def test_reward_events(self, profile):
        events = profile.reward_events
        assert "catch_pet" in events
        assert "win_battle" in events
        assert "defeat_world_boss" in events
        assert events["defeat_world_boss"] > events["win_battle"]
        assert events["battle_lost"] < 0
        assert events["pet_downed"] < 0

    def test_terminal_events(self, profile):
        assert "battle_lost" in profile.terminal_events
        for te in profile.terminal_events:
            assert te in profile.reward_events

    def test_state_classes_subset_of_reward_events(self, profile):
        for sc in profile.state_classes:
            assert sc in profile.reward_events, (
                f"State class '{sc}' not found in reward_events"
            )


class TestProfileRegistry:
    """Tests for the profile registry and factory."""

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

    def test_all_profiles_have_consistent_vocab(self):
        """Each profile's action_space should match its vocab_size."""
        for name, cls in PROFILES.items():
            profile = cls()
            space = profile.action_space
            assert space.vocab_size == profile.vocab_size, (
                f"{name}: action_space.vocab_size={space.vocab_size} "
                f"!= profile.vocab_size={profile.vocab_size}"
            )

    def test_all_profiles_have_complete_touch_mapping(self):
        """Every non-idle action should have a touch mapping."""
        for name, cls in PROFILES.items():
            profile = cls()
            for movement in profile.movements:
                if not profile.is_idle_movement(movement):
                    assert profile.get_touch_action(movement) is not None, (
                        f"{name}: missing touch mapping for movement '{movement}'"
                    )
            for action in profile.actions:
                if not profile.is_idle_action(action):
                    assert profile.get_touch_action(action) is not None, (
                        f"{name}: missing touch mapping for action '{action}'"
                    )
