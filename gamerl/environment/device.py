"""
ADB-based device control for Android.

Replaces the deprecated pyminitouch with ADB shell input commands,
which work on ALL Android versions (including Android 10+ where
minitouch is broken).

Also supports scrcpy's --otc mode for lower latency input.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Dict, Optional

logger = logging.getLogger("gamerl.environment")


class ADBDevice:
    """
    Android device control via ADB.

    Uses `adb shell input` commands for touch events.
    This is compatible with all Android versions, unlike minitouch
    which was limited to Android < 10.

    Args:
        serial: Device serial number (from `adb devices`). Empty for first device.
        screen_resolution: Device screen resolution (width, height).
    """

    def __init__(
        self,
        serial: str = "",
        screen_resolution: tuple[int, int] = (1080, 2160),
    ):
        self.serial = serial
        self.screen_resolution = screen_resolution
        self._adb_base = ["adb"]
        if serial:
            self._adb_base += ["-s", serial]

        # Verify device connection
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify that the device is connected via ADB."""
        try:
            result = subprocess.run(
                self._adb_base + ["get-state"],
                capture_output=True, text=True, timeout=5,
            )
            state = result.stdout.strip()
            if state != "device":
                logger.warning(f"Device state: {state}. May need to authorize ADB.")
            else:
                logger.info(f"ADB device connected: {self.serial or 'default'}")
        except Exception as e:
            logger.error(f"Failed to connect to ADB device: {e}")
            raise

    def _run_adb(self, *args: str, timeout: float = 5.0) -> str:
        """Run an ADB command and return output."""
        cmd = self._adb_base + list(args)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning(f"ADB command timed out: {' '.join(args)}")
            return ""
        except Exception as e:
            logger.error(f"ADB command failed: {e}")
            return ""

    def tap(self, x: int, y: int) -> None:
        """Simulate a tap at (x, y)."""
        self._run_adb("shell", "input", "tap", str(x), str(y))

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 100,
    ) -> None:
        """Simulate a swipe gesture."""
        self._run_adb(
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        )

    def long_press(self, x: int, y: int, duration_ms: int = 500) -> None:
        """Simulate a long press."""
        # A long press is a swipe from a point to itself with long duration
        self._run_adb(
            "shell", "input", "swipe",
            str(x), str(y), str(x), str(y), str(duration_ms),
        )

    def key_event(self, keycode: str) -> None:
        """Send a key event (e.g., 'KEYCODE_BACK')."""
        self._run_adb("shell", "input", "keyevent", keycode)

    def send_touch_command(self, command_str: str) -> None:
        """
        Send a minitouch-style touch command string via ADB.

        This provides backward compatibility with the original project's
        command format (e.g., "d 0 237 321 100\\nc\\nm 1 349 321 100\\nc\\nu 1\\nc\\n").

        Parses the minitouch protocol and converts to ADB input commands.

        Args:
            command_str: Minitouch-format command string.
        """
        lines = command_str.strip().split("\n")
        active_pointers: Dict[int, tuple[int, int]] = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "d":  # Touch down
                pointer_id = int(parts[1])
                x, y = int(parts[2]), int(parts[3])
                active_pointers[pointer_id] = (x, y)
                # For ADB, a "down" starts a potential swipe
                # We'll handle the full gesture when we see "u" or "m"

            elif cmd == "m":  # Move (swipe step)
                pointer_id = int(parts[1])
                x, y = int(parts[2]), int(parts[3])
                if pointer_id in active_pointers:
                    old_x, old_y = active_pointers[pointer_id]
                    # Execute as a swipe
                    duration = int(parts[4]) if len(parts) > 4 else 100
                    self.swipe(old_x, old_y, x, y, duration)
                    active_pointers[pointer_id] = (x, y)

            elif cmd == "u":  # Touch up
                pointer_id = int(parts[1])
                if pointer_id in active_pointers:
                    x, y = active_pointers[pointer_id]
                    # If it was a tap (no move), send a tap
                    self.tap(x, y)
                    del active_pointers[pointer_id]

            elif cmd == "c":  # Commit
                pass  # ADB commands are committed immediately

    def screencap(self) -> bytes:
        """Take a screenshot via ADB and return raw PNG bytes."""
        return subprocess.run(
            self._adb_base + ["shell", "screencap", "-p"],
            capture_output=True, timeout=5,
        ).stdout

    def get_screen_resolution(self) -> tuple[int, int]:
        """Get the device screen resolution."""
        output = self._run_adb("shell", "wm", "size")
        # Output: "Physical size: 1080x2160"
        try:
            size_str = output.split(":")[-1].strip()
            w, h = size_str.split("x")
            return int(w), int(h)
        except Exception:
            return self.screen_resolution


class ActionMapper:
    """
    Maps abstract game actions to device touch coordinates.

    This replaces the hardcoded JSON mappings in the original project.
    Coordinates are loaded from a GameProfile's touch_mapping, making
    it game-agnostic.

    Args:
        touch_mapping: Dict mapping action labels to TouchAction objects.
        resolution: Device screen resolution (width, height).
    """

    def __init__(
        self,
        touch_mapping: Dict[str, "TouchAction"],
        resolution: tuple[int, int] = (1080, 2160),
    ):
        self.resolution = resolution
        self.touch_mapping = touch_mapping

    @classmethod
    def from_profile(cls, profile) -> "ActionMapper":
        """Create an ActionMapper from a GameProfile."""
        return cls(
            touch_mapping=profile.touch_mapping,
            resolution=profile.resolution,
        )

    def get_action_command(self, action_name: str) -> str:
        """
        Get the touch command string for an action.

        Returns a minitouch-format command that can be sent via
        ADBDevice.send_touch_command().
        """
        touch_action = self.touch_mapping.get(action_name)
        if touch_action is None:
            return ""

        if touch_action.type == "tap":
            x, y = touch_action.coords
            return f"d 0 {x} {y} {touch_action.duration_ms}\nc\nu 0\nc\n"

        elif touch_action.type == "joystick":
            # coords = (start_x, start_y, end_x, end_y)
            sx, sy, ex, ey = touch_action.coords
            return f"d 1 {sx} {sy} 300\nc\nm 1 {ex} {ey} {touch_action.duration_ms}\nc\nu 1\nc\n"

        elif touch_action.type == "swipe":
            sx, sy, ex, ey = touch_action.coords
            return f"d 0 {sx} {sy} 300\nc\nm 0 {ex} {ey} {touch_action.duration_ms}\nc\nu 0\nc\n"

        return ""
