"""
OpenCV-based frame preprocessing pipeline.

Handles frame resize, normalization, ROI extraction, and color conversion
before feeding to the detection model.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("gamerl.vision")


class FramePreprocessor:
    """
    Preprocess raw screen captures for the detection pipeline.

    Pipeline:
        1. Convert PIL/RGB → BGR (OpenCV convention)
        2. Optionally crop to game area (remove letterbox/borders)
        3. Resize to model input size
        4. Normalize pixel values
        5. Optionally extract ROI crops for specific regions

    Args:
        input_size: Target (width, height) for the detection model.
        normalize: Whether to normalize to [0, 1].
        mean: Normalization mean (per channel).
        std: Normalization std (per channel).
        crop_box: Optional (left, top, right, bottom) to crop before resize.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.input_size = input_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.crop_box = crop_box

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for detection.

        Args:
            frame: Raw frame as numpy array (H, W, 3) in RGB or BGR.

        Returns:
            Preprocessed frame as (3, H, W) float32 tensor.
        """
        # Ensure BGR (OpenCV convention)
        if frame.shape[-1] == 3:
            # Assume RGB input, convert to BGR
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            img = frame

        # Crop if needed
        if self.crop_box is not None:
            l, t, r, b = self.crop_box
            img = img[t:b, l:r]

        # Resize
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        img = img.astype(np.float32)
        if self.normalize:
            img /= 255.0
            img = (img - self.mean) / self.std

        # HWC → CHW
        img = img.transpose(2, 0, 1)

        return img

    def preprocess_batch(self, frames: list[np.ndarray]) -> np.ndarray:
        """Preprocess a batch of frames."""
        return np.stack([self.preprocess(f) for f in frames])

    def extract_roi(
        self,
        frame: np.ndarray,
        regions: Dict[str, Tuple[int, int, int, int]],
    ) -> Dict[str, np.ndarray]:
        """
        Extract named regions of interest from a frame.

        Args:
            frame: Raw frame (H, W, 3).
            regions: Dict of name → (left, top, right, bottom).

        Returns:
            Dict of name → cropped image.
        """
        rois = {}
        for name, (l, t, r, b) in regions.items():
            rois[name] = frame[t:b, l:r].copy()
        return rois

    @staticmethod
    def resize_keep_ratio(
        frame: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Resize while keeping aspect ratio, padding with black.

        Returns:
            (resized_frame, scale, pad_w, pad_h)
        """
        h, w = frame.shape[:2]
        tw, th = target_size

        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = (tw - new_w) // 2
        pad_h = (th - new_h) // 2

        result = np.zeros((th, tw, frame.shape[2]), dtype=frame.dtype)
        result[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return result, scale, pad_w, pad_h
