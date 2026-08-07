"""Tests for the vision pipeline (detector + state builder + preprocessor)."""

import numpy as np
import pytest

from gamerl.vision.detector import (
    Detection,
    DetectionResult,
    MockDetector,
    create_detector,
)
from gamerl.vision.state_builder import (
    GameStateBuilder,
    StructuredState,
    EnemyState,
    TowerState,
)
from gamerl.vision.preprocess import FramePreprocessor


class TestDetection:
    """Test the Detection dataclass."""

    def test_detection_creation(self):
        det = Detection(
            class_name="hero",
            class_id=0,
            confidence=0.9,
            bbox=(100, 200, 300, 400),
        )
        assert det.class_name == "hero"
        assert det.confidence == 0.9
        assert det.center == (200.0, 300.0)
        assert det.width == 200.0
        assert det.height == 200.0
        assert det.area == 40000.0

    def test_to_dict(self):
        det = Detection("hero", 0, 0.9, (100, 200, 300, 400))
        d = det.to_dict()
        assert d["class_name"] == "hero"
        assert d["confidence"] == 0.9
        assert d["bbox"] == [100, 200, 300, 400]


class TestDetectionResult:
    """Test the DetectionResult container."""

    def _make_result(self):
        dets = [
            Detection("hero", 0, 0.9, (100, 100, 200, 200)),
            Detection("enemy", 1, 0.8, (300, 300, 400, 400)),
            Detection("enemy", 1, 0.6, (500, 500, 600, 600)),
            Detection("minion", 2, 0.3, (700, 700, 750, 750)),
        ]
        return DetectionResult(detections=dets, frame_shape=(800, 800))

    def test_filter_by_class(self):
        result = self._make_result()
        enemies = result.filter_by_class("enemy")
        assert len(enemies) == 2

    def test_filter_by_confidence(self):
        result = self._make_result()
        high_conf = result.filter_by_confidence(0.7)
        assert len(high_conf) == 2  # hero (0.9) and enemy (0.8)

    def test_get_highest_confidence(self):
        result = self._make_result()
        best = result.get_highest_confidence("enemy")
        assert best is not None
        assert best.confidence == 0.8

    def test_get_highest_confidence_none(self):
        result = self._make_result()
        best = result.get_highest_confidence("tower")
        assert best is None


class TestMockDetector:
    """Test the mock detector for testing without real models."""

    def test_detect_returns_result(self):
        detector = MockDetector(class_names=["hero", "enemy"], seed=42)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert isinstance(result, DetectionResult)
        assert result.frame_shape == (480, 640)
        assert result.inference_time_ms >= 0.0

    def test_detect_deterministic(self):
        """Same seed should produce same detections."""
        d1 = MockDetector(class_names=["hero", "enemy"], seed=42)
        d2 = MockDetector(class_names=["hero", "enemy"], seed=42)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        r1 = d1.detect(frame)
        r2 = d2.detect(frame)

        assert len(r1.detections) == len(r2.detections)
        for d1, d2 in zip(r1.detections, r2.detections):
            assert d1.bbox == d2.bbox
            assert d1.confidence == d2.confidence

    def test_detect_batch(self):
        detector = MockDetector(class_names=["hero", "enemy"], seed=42)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(frames)
        assert len(results) == 3

    def test_create_detector_mock(self):
        """Factory should create mock when no model path."""
        detector = create_detector(class_names=["hero", "enemy"])
        assert isinstance(detector, MockDetector)


class TestStructuredState:
    """Test the StructuredState dataclass."""

    def test_empty_state_vector(self):
        state = StructuredState()
        vec = state.to_vector()
        expected_dim = StructuredState.vector_dim()
        assert vec.shape == (expected_dim,)
        assert vec.dtype == np.float32

    def test_state_with_data(self):
        state = StructuredState(
            player_hp=0.75,
            player_pos=(0.3, 0.7),
            enemies=[
                EnemyState(pos=(0.5, 0.4), hp=0.3, distance=0.2),
                EnemyState(pos=(0.8, 0.6), hp=0.8, distance=0.5),
            ],
            skills_available=[1, 0, 1],
            towers=[TowerState(pos=(0.1, 0.9), team="enemy")],
            minions=[(0.2, 0.8), (0.3, 0.7)],
            gold=0.5,
        )
        vec = state.to_vector()
        assert vec.shape[0] == StructuredState.vector_dim()

        # Check player HP
        assert vec[0] == pytest.approx(0.75)
        # Check player pos
        assert vec[1] == pytest.approx(0.3)
        assert vec[2] == pytest.approx(0.7)

    def test_vector_dim_formula(self):
        dim = StructuredState.vector_dim(
            max_enemies=5, max_towers=6, max_minions=10, num_skills=3
        )
        # 1 (hp) + 2 (pos) + 5*4 (enemies) + 3 (skills) + 6*3 (towers) + 10*2 (minions) + 1 (gold)
        assert dim == 1 + 2 + 20 + 3 + 18 + 20 + 1

    def test_to_dict(self):
        state = StructuredState(player_hp=0.5, player_pos=(0.3, 0.7))
        d = state.to_dict()
        assert d["player_hp"] == 0.5
        assert d["player_pos"] == [0.3, 0.7]


class TestGameStateBuilder:
    """Test the GameStateBuilder."""

    def _make_builder(self):
        return GameStateBuilder(
            class_names=["hero", "enemy", "minion", "tower", "hp_bar", "skill_button"],
            screen_resolution=(1080, 2160),
            max_enemies=5,
            max_towers=6,
            max_minions=10,
            num_skills=3,
        )

    def test_build_empty_detections(self):
        builder = self._make_builder()
        result = DetectionResult(detections=[], frame_shape=(2160, 1080))
        state = builder.build(result)

        assert state.player_hp == 1.0  # Default when no HP bar
        assert state.num_detections == 0

    def test_build_with_player(self):
        builder = self._make_builder()
        dets = [
            Detection("hero", 0, 0.95, (500, 1800, 600, 1900)),
        ]
        result = DetectionResult(detections=dets, frame_shape=(2160, 1080))
        state = builder.build(result)

        # Player should be at center of bbox, normalized
        assert 0.4 < state.player_pos[0] < 0.6
        assert 0.8 < state.player_pos[1] < 0.9

    def test_build_with_enemies(self):
        builder = self._make_builder()
        dets = [
            Detection("hero", 0, 0.95, (500, 1800, 600, 1900)),  # player
            Detection("enemy", 1, 0.8, (200, 200, 250, 250)),    # enemy far
            Detection("enemy", 1, 0.7, (550, 1700, 600, 1750)),  # enemy near
        ]
        result = DetectionResult(detections=dets, frame_shape=(2160, 1080))
        state = builder.build(result)

        assert len(state.enemies) == 2
        # Nearest enemy should be first
        assert state.enemies[0].distance < state.enemies[1].distance

    def test_build_vector(self):
        builder = self._make_builder()
        result = DetectionResult(detections=[], frame_shape=(2160, 1080))
        vec = builder.build_vector(result)

        assert vec.dtype == np.float32
        assert vec.shape[0] == StructuredState.vector_dim(
            max_enemies=5, max_towers=6, max_minions=10, num_skills=3
        )

    def test_tower_team_heuristic(self):
        """Towers on top half should be 'enemy', bottom half 'ally'."""
        builder = self._make_builder()
        dets = [
            Detection("tower", 3, 0.9, (100, 100, 200, 300)),    # top → enemy
            Detection("tower", 3, 0.9, (100, 1800, 200, 2000)),  # bottom → ally
        ]
        result = DetectionResult(detections=dets, frame_shape=(2160, 1080))
        state = builder.build(result)

        assert len(state.towers) == 2
        teams = {t.team for t in state.towers}
        assert "enemy" in teams
        assert "ally" in teams


class TestFramePreprocessor:
    """Test the frame preprocessor."""

    def test_preprocess_shape(self):
        preprocessor = FramePreprocessor(input_size=(320, 320))
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.preprocess(frame)

        assert result.shape == (3, 320, 320)
        assert result.dtype == np.float32

    def test_preprocess_normalization(self):
        preprocessor = FramePreprocessor(input_size=(64, 64), normalize=True)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = preprocessor.preprocess(frame)

        # After normalization: (128/255 - mean) / std
        # Should not be all zeros
        assert np.any(result != 0)

    def test_preprocess_no_normalization(self):
        preprocessor = FramePreprocessor(input_size=(64, 64), normalize=False)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = preprocessor.preprocess(frame)

        # Without normalization, values stay as raw pixel values (0-255)
        assert np.allclose(result, 128.0, atol=0.1)

    def test_preprocess_batch(self):
        preprocessor = FramePreprocessor(input_size=(64, 64))
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        batch = preprocessor.preprocess_batch(frames)

        assert batch.shape == (3, 3, 64, 64)

    def test_extract_roi(self):
        preprocessor = FramePreprocessor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:40, 30:50] = 255  # White square

        rois = preprocessor.extract_roi(
            frame,
            {"test": (30, 20, 50, 40)},
        )

        assert "test" in rois
        assert rois["test"].shape == (20, 20, 3)
        assert np.all(rois["test"] == 255)

    def test_resize_keep_ratio(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result, scale, pad_w, pad_h = FramePreprocessor.resize_keep_ratio(
            frame, (100, 100)
        )

        assert result.shape == (100, 100, 3)
        assert scale == 0.5  # 100/200 = 0.5
