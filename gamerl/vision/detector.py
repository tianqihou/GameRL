"""
Game object detection using YOLO.

Supports two backends:
1. YOLODetector: Uses ultralytics YOLO for real detection (requires trained weights)
2. MockDetector: Returns random detections for testing without GPU/model

Detection categories are defined by the GameProfile's detection_classes property.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("gamerl.vision")


@dataclass
class Detection:
    """A single object detection result."""

    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    mask: Optional[np.ndarray] = None  # Optional segmentation mask

    @property
    def center(self) -> Tuple[float, float]:
        """Center of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": float(self.confidence),
            "bbox": list(self.bbox),
        }


@dataclass
class DetectionResult:
    """Full detection results for a frame."""

    detections: List[Detection] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (0, 0)  # (H, W)
    inference_time_ms: float = 0.0

    def filter_by_class(self, class_name: str) -> List[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.class_name == class_name]

    def filter_by_confidence(self, threshold: float) -> List[Detection]:
        """Get all detections above a confidence threshold."""
        return [d for d in self.detections if d.confidence >= threshold]

    def get_highest_confidence(self, class_name: str) -> Optional[Detection]:
        """Get the highest-confidence detection of a class."""
        filtered = self.filter_by_class(class_name)
        if not filtered:
            return None
        return max(filtered, key=lambda d: d.confidence)

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "frame_shape": list(self.frame_shape),
            "inference_time_ms": self.inference_time_ms,
        }


class GameDetector:
    """
    YOLO-based game object detector.

    Wraps ultralytics YOLO for real-time detection of game objects
    (heroes, HP bars, skill buttons, towers, minions).

    Args:
        model_path: Path to YOLO weights (.pt or .onnx).
        class_names: List of class names the model can detect.
        conf_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        device: Inference device ("cuda", "cpu", or "cuda:0").
        input_size: Model input size (width, height).
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.input_size = input_size
        self._model = None

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except ImportError:
            logger.warning(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a frame.

        Args:
            frame: Input frame (H, W, 3) in BGR format.

        Returns:
            DetectionResult with all detections.
        """
        import time

        start = time.perf_counter()

        results = self._model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size[0],
            device=self.device,
            verbose=False,
        )

        detections = []
        h, w = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()

                cls_name = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f"unknown_{cls_id}"
                )

                detections.append(
                    Detection(
                        class_name=cls_name,
                        class_id=cls_id,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            detections=detections,
            frame_shape=(h, w),
            inference_time_ms=elapsed_ms,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """Run detection on a batch of frames."""
        return [self.detect(f) for f in frames]


class MockDetector:
    """
    Mock detector for testing without a real model.

    Generates plausible-looking random detections that can be used
    to test the full pipeline without GPU or trained weights.
    """

    def __init__(
        self,
        class_names: List[str],
        conf_range: Tuple[float, float] = (0.5, 0.95),
        max_objects_per_class: int = 5,
        seed: Optional[int] = None,
    ):
        self.class_names = class_names
        self.conf_range = conf_range
        self.max_objects_per_class = max_objects_per_class
        self._rng = np.random.RandomState(seed)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Generate random detections."""
        import time

        start = time.perf_counter()
        h, w = frame.shape[:2]
        detections = []

        for cls_id, cls_name in enumerate(self.class_names):
            n_objects = self._rng.randint(0, self.max_objects_per_class + 1)

            for _ in range(n_objects):
                conf = self._rng.uniform(*self.conf_range)

                # Generate plausible bounding box
                box_w = self._rng.randint(30, min(150, w // 4))
                box_h = self._rng.randint(30, min(150, h // 4))
                x1 = self._rng.randint(0, w - box_w)
                y1 = self._rng.randint(0, h - box_h)
                x2 = x1 + box_w
                y2 = y1 + box_h

                detections.append(
                    Detection(
                        class_name=cls_name,
                        class_id=cls_id,
                        confidence=conf,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            detections=detections,
            frame_shape=(h, w),
            inference_time_ms=elapsed_ms,
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        return [self.detect(f) for f in frames]


def create_detector(
    model_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    device: str = "cuda",
    **kwargs,
) -> GameDetector | MockDetector:
    """
    Factory function to create a detector.

    If model_path is provided and ultralytics is installed, creates a
    GameDetector. Otherwise, creates a MockDetector for testing.

    Args:
        model_path: Path to YOLO weights, or None for mock.
        class_names: List of detectable class names.
        device: Inference device.
        **kwargs: Additional arguments passed to the detector.

    Returns:
        A detector instance (GameDetector or MockDetector).
    """
    if class_names is None:
        class_names = ["hero", "enemy", "minion", "tower", "hp_bar", "skill_button"]

    if model_path is None:
        logger.info("No model path provided, using MockDetector")
        return MockDetector(class_names=class_names, **kwargs)

    try:
        return GameDetector(
            model_path=model_path,
            class_names=class_names,
            device=device,
            **kwargs,
        )
    except ImportError:
        logger.warning("ultralytics not available, falling back to MockDetector")
        return MockDetector(class_names=class_names, **kwargs)
