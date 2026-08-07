"""
Runtime execution layer with dual-loop scheduling.

The dual-loop architecture separates high-frequency tactical control
from low-frequency strategic decision-making:

    ┌─────────────────────────────────────────┐
    │           SlowLoop (1 Hz)               │
    │  • Strategic planning (push/farm/fight) │
    │  • Target selection                     │
    │  • Item purchase decisions              │
    │  • VLM/LLM consultation (optional)      │
    └──────────────┬──────────────────────────┘
                   │ strategic directive
    ┌──────────────▼──────────────────────────┐
    │           FastLoop (30-100 Hz)          │
    │  • Movement and positioning             │
    │  • Skill casting and attacking           │
    │  • PPO policy execution                  │
    │  • Real-time dodge/react                 │
    └──────────────┬──────────────────────────┘
                   │ touch commands
                 ┌─▼─┐
                 │ADB│
                 └───┘
"""

from .latency_monitor import LatencyMonitor, LatencyStats
from .scheduler import (
    DualLoopScheduler,
    FastLoop,
    SlowLoop,
    StrategicDirective,
    LoopStatus,
)

__all__ = [
    "DualLoopScheduler",
    "FastLoop",
    "SlowLoop",
    "StrategicDirective",
    "LoopStatus",
    "LatencyMonitor",
    "LatencyStats",
]
