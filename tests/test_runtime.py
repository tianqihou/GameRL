"""Tests for the runtime dual-loop scheduler and latency monitor."""

import time
import numpy as np
import pytest

from gamerl.runtime.latency_monitor import LatencyMonitor, LatencyStats
from gamerl.runtime.scheduler import (
    DualLoopScheduler,
    FastLoop,
    SlowLoop,
    StrategicDirective,
    StrategicMode,
    LoopStatus,
)


class TestLatencyMonitor:
    """Test latency monitoring."""

    def test_record_and_stats(self):
        monitor = LatencyMonitor(window_size=100)
        for i in range(50):
            monitor.record("test", float(i))

        stats = monitor.get_stats("test")
        assert stats is not None
        assert stats.count == 50
        assert stats.min_ms == 0.0
        assert stats.max_ms == 49.0
        assert stats.mean_ms == pytest.approx(24.5)

    def test_start_end(self):
        monitor = LatencyMonitor()
        monitor.start("op")
        time.sleep(0.001)  # 1ms
        elapsed = monitor.end("op")

        assert elapsed > 0
        stats = monitor.get_stats("op")
        assert stats.count == 1

    def test_end_without_start(self):
        monitor = LatencyMonitor()
        elapsed = monitor.end("nonexistent")
        assert elapsed == 0.0

    def test_window_size(self):
        monitor = LatencyMonitor(window_size=5)
        for i in range(20):
            monitor.record("test", float(i))

        stats = monitor.get_stats("test")
        # Should only keep last 5 values: 15, 16, 17, 18, 19
        assert stats.count == 5
        assert stats.min_ms == 15.0

    def test_get_all_stats(self):
        monitor = LatencyMonitor()
        monitor.record("a", 1.0)
        monitor.record("b", 2.0)

        all_stats = monitor.get_all_stats()
        assert "a" in all_stats
        assert "b" in all_stats

    def test_total_latency(self):
        monitor = LatencyMonitor()
        monitor.record("capture", 10.0)
        monitor.record("detect", 20.0)
        monitor.record("decide", 5.0)

        total = monitor.get_total_latency()
        assert total.count == 3
        assert total.mean_ms == pytest.approx(11.67, rel=0.1)

    def test_reset(self):
        monitor = LatencyMonitor()
        monitor.record("test", 1.0)
        monitor.reset()

        assert monitor.get_stats("test") is None

    def test_stats_to_dict(self):
        monitor = LatencyMonitor()
        monitor.record("test", 5.0)

        stats = monitor.get_stats("test")
        d = stats.to_dict()
        assert d["name"] == "test"
        assert d["count"] == 1


class TestStrategicDirective:
    """Test strategic directive dataclass."""

    def test_default_directive(self):
        d = StrategicDirective()
        assert d.mode == StrategicMode.FARM
        assert d.aggression == 0.5
        assert d.urgency == 0.0

    def test_to_dict(self):
        d = StrategicDirective(
            mode=StrategicMode.RETREAT,
            target_pos=(0.5, 0.5),
            aggression=0.0,
            urgency=0.9,
            reasoning="Low HP",
        )
        result = d.to_dict()
        assert result["mode"] == "retreat"
        assert result["target_pos"] == [0.5, 0.5]
        assert result["urgency"] == 0.9


class TestStrategicMode:
    """Test strategic mode enum."""

    def test_all_modes(self):
        modes = list(StrategicMode)
        assert len(modes) == 8
        assert StrategicMode.FARM.value == "farm"
        assert StrategicMode.PUSH.value == "push"
        assert StrategicMode.TEAM_FIGHT.value == "fight"
        assert StrategicMode.RETREAT.value == "retreat"


class TestSlowLoop:
    """Test the slow (strategic) loop."""

    def test_default_heuristic_low_hp(self):
        """Low HP should trigger retreat."""
        loop = SlowLoop(target_hz=1.0)
        directive = loop.decide({"player_hp": 0.2, "enemies": [], "gold": 0.5})

        assert directive.mode == StrategicMode.RETREAT
        assert directive.urgency > 0.5
        assert directive.aggression == 0.0

    def test_default_heuristic_enemy_advantage(self):
        """High HP vs low HP enemy should trigger fight."""
        loop = SlowLoop(target_hz=1.0)
        directive = loop.decide({
            "player_hp": 0.8,
            "enemies": [{"hp": 0.3}],
            "gold": 0.5,
        })

        assert directive.mode == StrategicMode.TEAM_FIGHT
        assert directive.aggression > 0.5

    def test_default_heuristic_enemy_disadvantage(self):
        """Low HP vs enemy should trigger defend."""
        loop = SlowLoop(target_hz=1.0)
        directive = loop.decide({
            "player_hp": 0.4,
            "enemies": [{"hp": 0.8}],
            "gold": 0.5,
        })

        assert directive.mode == StrategicMode.DEFEND
        assert directive.aggression < 0.5

    def test_default_heuristic_farm(self):
        """No enemies, low gold should trigger farm."""
        loop = SlowLoop(target_hz=1.0)
        directive = loop.decide({"player_hp": 1.0, "enemies": [], "gold": 0.1})

        assert directive.mode == StrategicMode.FARM

    def test_default_heuristic_push(self):
        """No enemies, high gold should trigger push."""
        loop = SlowLoop(target_hz=1.0)
        directive = loop.decide({"player_hp": 1.0, "enemies": [], "gold": 0.5})

        assert directive.mode == StrategicMode.PUSH

    def test_custom_heuristic(self):
        """Custom heuristic callback should be used."""
        def custom_heuristic(state):
            return StrategicDirective(
                mode=StrategicMode.AMBUSH,
                reasoning="Custom strategy",
            )

        loop = SlowLoop(heuristic_callback=custom_heuristic)
        directive = loop.decide({})

        assert directive.mode == StrategicMode.AMBUSH
        assert directive.reasoning == "Custom strategy"

    def test_vlm_callback(self):
        """VLM callback should be used when provided."""
        def vlm_consult(state):
            return StrategicDirective(
                mode=StrategicMode.ROTATE,
                reasoning="VLM says rotate to mid lane",
            )

        loop = SlowLoop(vlm_callback=vlm_consult)
        directive = loop.decide({})

        assert directive.mode == StrategicMode.ROTATE

    def test_error_fallback(self):
        """Errors in callback should fall back to default heuristic."""
        def bad_callback(state):
            raise ValueError("VLM API error")

        loop = SlowLoop(vlm_callback=bad_callback)
        directive = loop.decide({"player_hp": 1.0, "enemies": []})

        # Should fall back to default heuristic
        assert directive.mode in [StrategicMode.FARM, StrategicMode.PUSH]
        assert loop.status.error_count == 1

    def test_get_directive(self):
        loop = SlowLoop()
        initial = loop.get_directive()
        assert initial.mode == StrategicMode.FARM  # Default

        loop.decide({"player_hp": 0.1, "enemies": []})
        updated = loop.get_directive()
        assert updated.mode == StrategicMode.RETREAT


class TestLoopStatus:
    """Test loop status tracking."""

    def test_default_status(self):
        status = LoopStatus()
        assert status.iteration == 0
        assert status.running is False
        assert status.error_count == 0


class TestDualLoopScheduler:
    """Test the dual-loop scheduler coordination."""

    def test_create_scheduler(self):
        fast = FastLoop(agent=None, env=None, target_hz=30.0)
        slow = SlowLoop(target_hz=1.0)
        scheduler = DualLoopScheduler(fast, slow)

        status = scheduler.get_status()
        assert status["running"] is False
        assert "fast_loop" in status
        assert "slow_loop" in status
        assert "directive" in status

    def test_sync_directive(self):
        fast = FastLoop(agent=None, env=None, target_hz=30.0)
        slow = SlowLoop(target_hz=1.0)

        # Set a directive in slow loop
        slow.decide({"player_hp": 0.1, "enemies": []})
        assert slow.get_directive().mode == StrategicMode.RETREAT

        scheduler = DualLoopScheduler(fast, slow)
        directive = scheduler.sync_directive()

        assert directive.mode == StrategicMode.RETREAT
        assert fast.get_directive().mode == StrategicMode.RETREAT

    def test_default_state_fn(self):
        fast = FastLoop(agent=None, env=None, target_hz=30.0)
        slow = SlowLoop(target_hz=1.0)
        scheduler = DualLoopScheduler(fast, slow)

        state = scheduler._default_state_fn()
        assert state["player_hp"] == 1.0
        assert state["enemies"] == []
