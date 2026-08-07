"""
Cross-platform screen capture.

Replaces the Windows-only win32gui-based screenshot in the original project.
Supports multiple backends:
- mss: Fast cross-platform screen capture (recommended)
- win32: Windows-specific win32gui capture (backward compatible)
- pyqt5: Qt-based capture (for scrcpy windows)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("gamerl.environment")

# Try importing optional backends
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import win32gui, win32ui, win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QImage
    HAS_PYQT5 = True
except ImportError:
    HAS_PYQT5 = False


class ScreenCapture:
    """
    Screen capture with pluggable backends.

    Args:
        method: Capture method - "mss", "win32", or "pyqt5".
        window_title: Window title to capture (for win32/pyqt5).
        target_size: Target (width, height) for the captured image.
        crop_box: Optional (left, top, right, bottom) crop box.
        monitor_index: Monitor index for mss (0 = primary).
    """

    def __init__(
        self,
        method: str = "mss",
        window_title: str = "scrcpy",
        target_size: Tuple[int, int] = (960, 540),
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        monitor_index: int = 0,
    ):
        self.method = method
        self.window_title = window_title
        self.target_size = target_size
        self.crop_box = crop_box
        self.monitor_index = monitor_index

        self._sct = None  # mss instance
        self._app = None  # PyQt5 app
        self._hwnd = None  # win32 window handle

        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize the selected capture backend."""
        if self.method == "mss":
            if not HAS_MSS:
                raise ImportError("mss is not installed. Install with: pip install mss")
            self._sct = mss.mss()
            logger.info("Screen capture: mss backend initialized")

        elif self.method == "win32":
            if not HAS_WIN32:
                raise ImportError("pywin32 is not installed. Install with: pip install pywin32")
            self._hwnd = win32gui.FindWindow(0, self.window_title)
            if not self._hwnd:
                raise RuntimeError(f"Window '{self.window_title}' not found")
            logger.info(f"Screen capture: win32 backend (window='{self.window_title}')")

        elif self.method == "pyqt5":
            if not HAS_PYQT5:
                raise ImportError("PyQt5 is not installed. Install with: pip install PyQt5")
            if not self._app:
                self._app = QApplication.instance() or QApplication(sys.argv)
            self._hwnd = win32gui.FindWindow(0, self.window_title) if HAS_WIN32 else None
            logger.info(f"Screen capture: pyqt5 backend (window='{self.window_title}')")

        else:
            raise ValueError(f"Unknown capture method: {self.method}")

    def capture(self) -> Image.Image:
        """
        Capture a screenshot.

        Returns:
            PIL Image of the captured screen, resized to target_size.
        """
        if self.method == "mss":
            return self._capture_mss()
        elif self.method == "win32":
            return self._capture_win32()
        elif self.method == "pyqt5":
            return self._capture_pyqt5()
        else:
            raise RuntimeError(f"Backend {self.method} not initialized")

    def _capture_mss(self) -> Image.Image:
        """Capture using mss (cross-platform, fast)."""
        monitor = self._sct.monitors[self.monitor_index]
        raw = self._sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        if self.crop_box:
            img = img.crop(self.crop_box)

        img = img.resize(self.target_size, Image.BILINEAR)
        return img

    def _capture_win32(self) -> Image.Image:
        """Capture using win32gui (Windows-specific)."""
        left, top, right, bot = win32gui.GetWindowRect(self._hwnd)
        width = right - left
        height = bot - top

        hwnd_dc = win32gui.GetWindowDC(self._hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)
        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
        )

        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self._hwnd, hwnd_dc)

        if self.crop_box:
            img = img.crop(self.crop_box)

        img = img.resize(self.target_size, Image.BILINEAR)
        return img

    def _capture_pyqt5(self) -> Image.Image:
        """Capture using PyQt5 (for scrcpy windows)."""
        screen = self._app.primaryScreen()
        if self._hwnd and HAS_WIN32:
            img_qt = screen.grabWindow(self._hwnd)
        else:
            img_qt = screen.grabWindow(0)

        # Convert QImage to PIL Image
        img_qt = img_qt.convertToFormat(QImage.Format.Format_RGB888)
        width = img_qt.width()
        height = img_qt.height()
        ptr = img_qt.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        img = Image.fromarray(arr)

        if self.crop_box:
            img = img.crop(self.crop_box)

        img = img.resize(self.target_size, Image.BILINEAR)
        return img

    def close(self) -> None:
        """Clean up resources."""
        if self._sct:
            self._sct.close()
            self._sct = None
