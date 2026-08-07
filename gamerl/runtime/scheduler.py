"""
Dual-loop scheduler for hierarchical game AI control.

Architecture:

    SlowLoop (1 Hz)                 FastLoop (30-100 Hz)
    ┌────────────────┐              ┌────────────────────┐
    │ Strategic      │  directive   │ Tactical           │
    │ • push/farm    │─────────────▶│ • PPO policy       │
    │ • team fight   │              │ • movement         │
    │ • retreat      │              │ • skill cast       │
    │ • VLM consult  │              │ • attack           │
    └────────────────┘              └─────────┬──────────┘
                                             │ actions
                                             ▼
                                         ADB Device

The SlowLoop runs at low frequency and makes high-level strategic decisions.
It does NOT block the FastLoop. The FastLoop runs continuously at high
frequency, executing the PPO policy with the current strategic directive
as context.

The strategic directive can bias action selection (e.g., if directive says
"retreat", the FastLoop adds a negative reward for advancing).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

import numpy as np

from .latency_monitor import LatencyMonitor

logger = logging.getLogger("gamerl.runtime")


class StrategicMode(Enum):
    """High-level strategic modes."""

    FARM = "farm"           # Focus on killing minions for gold
    PUSH = "push"           # Push lanes and destroy towers
    TEAM_FIGHT = "fight"    # Engage in team fights
    RETREAT = "retreat"     # Fall back to safety
    DEFEND = "defend"       # Defend own towers/base
    AMBUSH = "ambush"       # Hide and wait for enemies
    ROTATE = "rotate"       # Move to another lane
    JUNGLE = "jungle"       # Farm jungle monsters


@dataclass
class StrategicDirective:
    """
    A strategic directive from the SlowLoop to the FastLoop.

    The FastLoop uses this to bias action selection and reward shaping.
    """

    mode: StrategicMode = StrategicMode.FARM
    target_pos: Optional[tuple[float, float]] = None  # Where to go (normalized)
    target_class: Optional[str] = None  # What to target ("enemy", "tower", "minion")
    aggression: float = 0.5  # 0=defensive, 1=aggressive
    urgency: float = 0.0  # 0=relaxed, 1=urgent (e.g., low HP)
    reasoning: str = ""  # Human-readable explanation

    # Reward shaping weights for the FastLoop
    reward_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "target_pos": list(self.target_pos) if self.target_pos else None,
            "target_class": self.target_class,
            "aggression": self.aggression,
            "urgency": self.urgency,
            "reasoning": self.reasoning,
        }


@dataclass
class LoopStatus:
    """Status of a loop iteration."""

    iteration: int = 0
    last_cycle_time_ms: float = 0.0
    target_hz: float = 0.0
    actual_hz: float = 0.0
    running: bool = False
    error_count: int = 0
    last_error: Optional[str] = None


class FastLoop:
    """
    High-frequency tactical control loop (30-100 Hz).

    Runs the PPO policy at high frequency for real-time movement,
    skill casting, and attacking. The strategic directive from the
    SlowLoop biases action selection and reward shaping.

    Args:
        agent: PPO agent for action selection.
        env: Game environment for state/action execution.
        target_hz: Target frequency (default 30 Hz).
    """

    def __init__(
        self,
        agent,
        env,
        target_hz: float = 30.0,
    ):
        self.agent = agent
        self.env = env
        self.target_hz = target_hz
        self._interval = 1.0 / target_hz

        self.status = LoopStatus(target_hz=target_hz)
        self._current_directive = StrategicDirective()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.latency = LatencyMonitor(window_size=200)

    def set_directive(self, directive: StrategicDirective) -> None:
        """Update the strategic directive (thread-safe)."""
        self._current_directive = directive
        logger.debug(f"FastLoop directive updated: {directive.mode.value}")

    def get_directive(self) -> StrategicDirective:
        """Get the current strategic directive."""
        return self._current_directive

    def run_once(self) -> Optional[tuple]:
        """
        Execute one cycle of the fast loop.

        Returns:
            Tuple of (action, log_prob, value) or None on error.
        """
        start = time.perf_counter()
        self.status.iteration += 1

        try:
            # 1. Get current state
            self.latency.start("state_capture")
            state = self.env.reset() if self.status.iteration == 1 else self.env._get_state()
            self.latency.end("state_capture")

            # 2. Select action via PPO
            self.latency.start("decide")
            action, log_prob, value = self.agent.select_action(
                state.image_features,
                state.action_history,
            )
            self.latency.end("decide")

            # 3. Execute action
            self.latency.start("execute")
            self.env.step(action)
            self.latency.end("execute")

            # 4. Store transition for later PPO update
            self.agent.store_transition(
                state.image_features,
                action,
                log_prob,
                value,
                reward=0.0,  # Reward computed by state judgment model
                done=False,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.status.last_cycle_time_ms = elapsed_ms
            self.status.actual_hz = 1000.0 / max(elapsed_ms, 0.1)

            return action, log_prob, value

        except Exception as e:
            self.status.error_count += 1
            self.status.last_error = str(e)
            logger.error(f"FastLoop error (iter {self.status.iteration}): {e}")
            return None

    def start(self) -> None:
        """Start the fast loop in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self.status.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"FastLoop started at {self.target_hz} Hz")

    def stop(self) -> None:
        """Stop the fast loop."""
        self._stop_event.set()
        self.status.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("FastLoop stopped")

    def _run(self) -> None:
        """Main loop thread."""
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()

            self.run_once()

            # Sleep to maintain target frequency
            elapsed = time.perf_counter() - cycle_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class SlowLoop:
    """
    Low-frequency strategic decision loop (1 Hz).

    Analyzes the game state at low frequency and produces strategic
    directives for the FastLoop. Optionally consults a VLM/LLM for
    high-level reasoning.

    Args:
        target_hz: Target frequency (default 1 Hz).
        vlm_callback: Optional function(state_dict) -> StrategicDirective
            for VLM-based strategic reasoning.
        heuristic_callback: Optional function(state_dict) -> StrategicDirective
            for rule-based strategic reasoning.
    """

    def __init__(
        self,
        target_hz: float = 1.0,
        vlm_callback: Optional[Callable] = None,
        heuristic_callback: Optional[Callable] = None,
    ):
        self.target_hz = target_hz
        self._interval = 1.0 / target_hz
        self.vlm_callback = vlm_callback
        self.heuristic_callback = heuristic_callback or self._default_heuristic

        self.status = LoopStatus(target_hz=target_hz)
        self._current_directive = StrategicDirective()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.latency = LatencyMonitor(window_size=50)

    def get_directive(self) -> StrategicDirective:
        """Get the current strategic directive."""
        return self._current_directive

    def decide(self, state_dict: dict) -> StrategicDirective:
        """
        Make a strategic decision.

        Priority: VLM callback > heuristic callback > default.

        Args:
            state_dict: Current game state as a dictionary.

        Returns:
            StrategicDirective for the FastLoop.
        """
        start = time.perf_counter()

        try:
            if self.vlm_callback is not None:
                self.latency.start("vlm_consult")
                directive = self.vlm_callback(state_dict)
                self.latency.end("vlm_consult")
            else:
                self.latency.start("heuristic")
                directive = self.heuristic_callback(state_dict)
                self.latency.end("heuristic")

            if directive is None:
                directive = self._default_heuristic(state_dict)

            self._current_directive = directive

        except Exception as e:
            logger.error(f"SlowLoop error: {e}, using default directive")
            self.status.error_count += 1
            self.status.last_error = str(e)
            directive = self._default_heuristic(state_dict)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.status.last_cycle_time_ms = elapsed_ms
        self.status.iteration += 1

        return directive

    def start(self, get_state_fn: Callable[[], dict]) -> None:
        """
        Start the slow loop in a background thread.

        Args:
            get_state_fn: Function that returns the current game state dict.
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self.status.running = True
        self._thread = threading.Thread(
            target=self._run, args=(get_state_fn,), daemon=True
        )
        self._thread.start()
        logger.info(f"SlowLoop started at {self.target_hz} Hz")

    def stop(self) -> None:
        """Stop the slow loop."""
        self._stop_event.set()
        self.status.running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("SlowLoop stopped")

    def _run(self, get_state_fn: Callable[[], dict]) -> None:
        """Main loop thread."""
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()

            state_dict = get_state_fn()
            self.decide(state_dict)

            elapsed = time.perf_counter() - cycle_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _default_heuristic(self, state_dict: dict) -> StrategicDirective:
        """
        Default rule-based strategic reasoning.

        Uses simple heuristics based on game state:
        - Low HP → retreat
        - Enemies nearby with high HP → team fight or retreat
        - No enemies → farm/push
        """
        player_hp = state_dict.get("player_hp", 1.0)
        enemies = state_dict.get("enemies", [])
        player_gold = state_dict.get("gold", 0.0)

        # Low HP → retreat
        if player_hp < 0.3:
            return StrategicDirective(
                mode=StrategicMode.RETREAT,
                urgency=1.0 - player_hp,
                aggression=0.0,
                reasoning=f"Low HP ({player_hp:.0%}), retreating to safety",
                reward_weights={"advance": -1.0, "retreat": 2.0},
            )

        # Enemies nearby with high HP → fight if we have HP advantage
        if enemies:
            enemy_count = len(enemies)
            avg_enemy_hp = sum(e.get("hp", 1.0) for e in enemies) / enemy_count

            if player_hp > 0.6 and avg_enemy_hp < player_hp:
                return StrategicDirective(
                    mode=StrategicMode.TEAM_FIGHT,
                    target_class="enemy",
                    aggression=0.8,
                    urgency=0.3,
                    reasoning=f"Engaging {enemy_count} enemies (HP advantage)",
                    reward_weights={"engage": 1.5, "kill": 3.0},
                )
            else:
                return StrategicDirective(
                    mode=StrategicMode.DEFEND,
                    target_class="enemy",
                    aggression=0.3,
                    urgency=0.5,
                    reasoning=f"Defensive play vs {enemy_count} enemies",
                    reward_weights={"engage": -0.5, "positioning": 1.0},
                )

        # No enemies → farm or push
        if player_gold < 0.3:
            return StrategicDirective(
                mode=StrategicMode.FARM,
                target_class="minion",
                aggression=0.4,
                reasoning="No enemies nearby, farming minions for gold",
                reward_weights={"kill_minion": 2.0},
            )
        else:
            return StrategicDirective(
                mode=StrategicMode.PUSH,
                target_class="tower",
                aggression=0.6,
                reasoning="No enemies nearby, pushing lane",
                reward_weights={"push_tower": 3.0, "advance": 0.5},
            )


class DualLoopScheduler:
    """
    Coordinates the FastLoop and SlowLoop.

    The scheduler manages the lifecycle of both loops and provides
    a unified interface for starting/stopping the game AI.

    Args:
        fast_loop: The fast (tactical) loop.
        slow_loop: The slow (strategic) loop.
        get_state_fn: Function to get current game state as dict.
    """

    def __init__(
        self,
        fast_loop: FastLoop,
        slow_loop: SlowLoop,
        get_state_fn: Optional[Callable[[], dict]] = None,
    ):
        self.fast_loop = fast_loop
        self.slow_loop = slow_loop
        self.get_state_fn = get_state_fn or self._default_state_fn
        self._running = False

    def start(self) -> None:
        """Start both loops."""
        if self._running:
            return

        logger.info("Starting DualLoopScheduler...")
        self._running = True

        # Start slow loop first (it produces directives for fast loop)
        self.slow_loop.start(self.get_state_fn)

        # Small delay to let slow loop produce initial directive
        time.sleep(0.1)

        # Start fast loop
        self.fast_loop.start()

        logger.info("DualLoopScheduler running")

    def stop(self) -> None:
        """Stop both loops."""
        if not self._running:
            return

        logger.info("Stopping DualLoopScheduler...")
        self._running = False

        self.fast_loop.stop()
        self.slow_loop.stop()

        logger.info("DualLoopScheduler stopped")

    def sync_directive(self) -> StrategicDirective:
        """
        Sync the strategic directive from SlowLoop to FastLoop.

        Call this periodically (or it happens automatically in the loops)
        to ensure the FastLoop has the latest directive.
        """
        directive = self.slow_loop.get_directive()
        self.fast_loop.set_directive(directive)
        return directive

    def get_status(self) -> dict:
        """Get status of both loops."""
        return {
            "running": self._running,
            "fast_loop": {
                "iteration": self.fast_loop.status.iteration,
                "actual_hz": round(self.fast_loop.status.actual_hz, 1),
                "errors": self.fast_loop.status.error_count,
            },
            "slow_loop": {
                "iteration": self.slow_loop.status.iteration,
                "actual_hz": round(
                    1000.0 / max(self.slow_loop.status.last_cycle_time_ms, 0.1), 1
                ),
                "errors": self.slow_loop.status.error_count,
            },
            "directive": self.slow_loop.get_directive().to_dict(),
        }

    def log_latency(self) -> None:
        """Log latency summary for both loops."""
        logger.info("=== FastLoop Latency ===")
        self.fast_loop.latency.log_summary()

        logger.info("=== SlowLoop Latency ===")
        self.slow_loop.latency.log_summary()

    def _default_state_fn(self) -> dict:
        """Default state function (returns empty state)."""
        return {
            "player_hp": 1.0,
            "player_pos": [0.5, 0.5],
            "enemies": [],
            "gold": 0.0,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
