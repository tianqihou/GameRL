"""
Latency monitoring for real-time game AI.

Tracks timing for each stage of the perception-action loop:
    Capture → Detect → Decide → Execute

Provides rolling statistics and alerts when latency exceeds thresholds.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

logger = logging.getLogger("gamerl.runtime")


@dataclass
class LatencyStats:
    """Statistics for a single timing channel."""

    name: str
    count: int = 0
    mean_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


class LatencyMonitor:
    """
    Monitor latency across pipeline stages.

    Usage:
        monitor = LatencyMonitor(window_size=500)
        monitor.start("capture")
        ... do capture ...
        monitor.end("capture")

        stats = monitor.get_stats("capture")
        monitor.log_summary()

    Args:
        window_size: Number of recent samples to keep per channel.
        alert_threshold_ms: Alert if any channel exceeds this mean.
    """

    def __init__(
        self,
        window_size: int = 500,
        alert_threshold_ms: float = 50.0,
    ):
        self.window_size = window_size
        self.alert_threshold_ms = alert_threshold_ms

        self._channels: Dict[str, Deque[float]] = {}
        self._active: Dict[str, float] = {}  # name → start timestamp

    def start(self, name: str) -> None:
        """Start timing a channel."""
        self._active[name] = time.perf_counter()

    def end(self, name: str) -> float:
        """
        End timing a channel and record the elapsed time.

        Returns:
            Elapsed time in milliseconds.
        """
        if name not in self._active:
            logger.warning(f"LatencyMonitor.end('{name}') without start")
            return 0.0

        elapsed_ms = (time.perf_counter() - self._active.pop(name)) * 1000
        self.record(name, elapsed_ms)
        return elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing sample directly (without start/end)."""
        if name not in self._channels:
            self._channels[name] = deque(maxlen=self.window_size)
        self._channels[name].append(elapsed_ms)

    def get_stats(self, name: str) -> Optional[LatencyStats]:
        """Get statistics for a channel."""
        if name not in self._channels or not self._channels[name]:
            return None

        samples = sorted(self._channels[name])
        n = len(samples)

        stats = LatencyStats(name=name)
        stats.count = n
        stats.mean_ms = sum(samples) / n
        stats.min_ms = samples[0]
        stats.max_ms = samples[-1]
        stats.p50_ms = samples[n // 2]
        stats.p95_ms = samples[int(n * 0.95)] if n > 1 else samples[0]
        stats.p99_ms = samples[int(n * 0.99)] if n > 1 else samples[0]

        return stats

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all channels."""
        return {name: self.get_stats(name) for name in self._channels}

    def get_total_latency(self) -> LatencyStats:
        """Get combined latency across all channels."""
        all_samples: list[float] = []
        for samples in self._channels.values():
            all_samples.extend(samples)

        if not all_samples:
            return LatencyStats(name="total")

        all_samples.sort()
        n = len(all_samples)
        stats = LatencyStats(name="total")
        stats.count = n
        stats.mean_ms = sum(all_samples) / n
        stats.min_ms = all_samples[0]
        stats.max_ms = all_samples[-1]
        stats.p50_ms = all_samples[n // 2]
        stats.p95_ms = all_samples[int(n * 0.95)]
        stats.p99_ms = all_samples[int(n * 0.99)]
        return stats

    def log_summary(self) -> None:
        """Log a summary of all latency channels."""
        for name in self._channels:
            stats = self.get_stats(name)
            if stats is None:
                continue

            level = logging.WARNING if stats.mean_ms > self.alert_threshold_ms else logging.INFO
            logger.log(
                level,
                f"[{name}] mean={stats.mean_ms:.1f}ms "
                f"p50={stats.p50_ms:.1f}ms p95={stats.p95_ms:.1f}ms "
                f"p99={stats.p99_ms:.1f}ms (n={stats.count})",
            )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._channels.clear()
        self._active.clear()
