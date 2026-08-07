"""
Logging setup with structured output and TensorBoard integration.

Replaces the scattered `print()` calls in the original project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def setup_logger(
    name: str = "gamerl",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """TensorBoard metrics logger for training visualization."""

    def __init__(self, log_dir: str | Path):
        """
        Initialize the metrics logger.

        Args:
            log_dir: Directory for TensorBoard logs.
        """
        self.log_dir = str(log_dir)
        self._writer: Optional["SummaryWriter"] = None

    @property
    def writer(self) -> Optional["SummaryWriter"]:
        """Lazy-initialize the TensorBoard writer."""
        if self._writer is None and HAS_TENSORBOARD:
            self._writer = SummaryWriter(self.log_dir)
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram of values."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        if self.writer:
            self.writer.add_text(tag, text, step)

    def close(self) -> None:
        """Close the writer."""
        if self._writer:
            self._writer.close()
            self._writer = None
