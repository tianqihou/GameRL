"""
TensorRT engine building and inference.

Provides:
- build_trt_engine: Convert ONNX → TensorRT engine with FP16/INT8
- TRTEngine: Load and run TensorRT engines
- TRTInferencer: High-level inference wrapper with pre/post processing

TensorRT reduces inference latency significantly:
    PyTorch (FP32):  ~40ms per inference
    TensorRT (FP16): ~15ms per inference
    TensorRT (INT8): ~8ms per inference

Requirements:
    pip install tensorrt (or use the TensorRT wheel from NVIDIA)
    NVIDIA GPU with CUDA support
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("gamerl.inference")


@dataclass
class TRTConfig:
    """Configuration for TensorRT engine building."""

    fp16: bool = True
    int8: bool = False
    max_batch_size: int = 32
    max_workspace_size: int = 4 << 30  # 4 GB
    min_timing_iterations: int = 1
    avg_timing_iterations: int = 8

    # INT8 calibration (required if int8=True)
    calib_data: Optional[np.ndarray] = None  # Calibration dataset
    calib_cache: Optional[str] = None  # Path to calibration cache file


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    config: Optional[TRTConfig] = None,
) -> Path:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Path to save the TensorRT engine.
        config: Build configuration. Uses defaults if None.

    Returns:
        Path to the saved engine file.

    Raises:
        ImportError: If TensorRT is not installed.
        RuntimeError: If engine building fails.
    """
    if config is None:
        config = TRTConfig()

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT is not installed. Install it from "
            "https://developer.nvidia.com/tensorrt"
        )

    logger.info(f"Building TensorRT engine from {onnx_path}")
    logger.info(f"  FP16: {config.fp16}, INT8: {config.int8}")
    logger.info(f"  Max batch size: {config.max_batch_size}")

    # Create TRT logger
    trt_logger = trt.Logger(trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(trt_logger)

    # Create network
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # Parse ONNX
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config_builder = builder.create_builder_config()
    config_builder.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, config.max_workspace_size
    )

    if config.fp16:
        config_builder.set_flag(trt.BuilderFlag.FP16)
        logger.info("  Enabled FP16 precision")

    if config.int8:
        config_builder.set_flag(trt.BuilderFlag.INT8)
        logger.info("  Enabled INT8 precision")

        # Set up INT8 calibrator if data provided
        if config.calib_data is not None:
            calibrator = _Int8Calibrator(
                config.calib_data,
                cache_file=config.calib_cache,
            )
            config_builder.int8_calibrator = calibrator
        else:
            logger.warning(
                "INT8 enabled without calibration data. "
                "This may result in poor accuracy."
            )

    # Optimization profiles for dynamic batch
    profile = builder.create_optimization_profile()

    # Get input names and set dynamic shapes
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape

        # If batch dim is dynamic (-1), set profile
        if shape[0] == -1:
            min_shape = list(shape)
            min_shape[0] = 1
            opt_shape = list(shape)
            opt_shape[0] = config.max_batch_size // 2
            max_shape = list(shape)
            max_shape[0] = config.max_batch_size

            # Fill other dynamic dims with reasonable defaults
            for j in range(1, len(min_shape)):
                if min_shape[j] == -1:
                    min_shape[j] = 1
                    opt_shape[j] = 10
                    max_shape[j] = 300

            profile.set_shape(
                name,
                min=tuple(min_shape),
                opt=tuple(opt_shape),
                max=tuple(max_shape),
            )

    config_builder.add_optimization_profile(profile)

    # Build engine
    logger.info("Building engine (this may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config_builder)

    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized)

    file_size_mb = engine_path.stat().st_size / (1024 * 1024)
    logger.info(f"TensorRT engine saved: {engine_path} ({file_size_mb:.1f} MB)")

    return engine_path


class TRTEngine:
    """
    Load and run a TensorRT engine for inference.

    Args:
        engine_path: Path to the TensorRT engine file.
        device_id: GPU device ID.
    """

    def __init__(
        self,
        engine_path: str | Path,
        device_id: int = 0,
    ):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError(
                "TensorRT and PyCUDA are required. Install them from "
                "https://developer.nvidia.com/tensorrt"
            )

        self.trt = trt
        self.cuda = cuda
        self.device_id = device_id

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        logger.info(f"Loading TensorRT engine: {engine_path}")

        # Load engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._setup_io_buffers()

        logger.info(
            f"Engine loaded. Inputs: {list(self.input_shapes.keys())}, "
            f"Outputs: {list(self.output_shapes.keys())}"
        )

    def _setup_io_buffers(self) -> None:
        """Set up input/output buffer mappings."""
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = self.engine.get_tensor_shape(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = self.engine.get_tensor_shape(name)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the engine.

        Args:
            inputs: Dict mapping input names to numpy arrays.

        Returns:
            Dict mapping output names to numpy arrays.
        """
        import pycuda.driver as cuda

        # Set dynamic shapes
        for name, arr in inputs.items():
            if name in self.input_names:
                self.context.set_input_shape(name, arr.shape)

        # Allocate device memory
        bindings = []
        input_buffers = {}
        output_buffers = {}

        for name in self.input_names:
            arr = inputs[name]
            shape = tuple(arr.shape)
            size = int(np.prod(shape))
            dtype = arr.dtype

            # Allocate device memory
            d_mem = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(d_mem, np.ascontiguousarray(arr))
            bindings.append(int(d_mem))
            input_buffers[name] = (d_mem, shape, dtype)

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            size = int(np.prod(shape))
            dtype = np.float32  # TRT outputs are typically float32

            d_mem = cuda.mem_alloc(size * 4)
            bindings.append(int(d_mem))
            output_buffers[name] = (d_mem, shape, dtype)

        # Execute
        self.context.execute_async_v2(bindings, 0)

        # Copy outputs back
        results = {}
        for name, (d_mem, shape, dtype) in output_buffers.items():
            arr = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(arr, d_mem)
            results[name] = arr

        # Free device memory
        for d_mem, _, _ in list(input_buffers.values()) + list(output_buffers.values()):
            d_mem.free()

        return results

    def __del__(self):
        """Clean up."""
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine


class TRTInferencer:
    """
    High-level TensorRT inference wrapper.

    Handles pre/post processing for the policy network:
    - Pre: Convert numpy state to TRT input format
    - Infer: Run the TRT engine
    - Post: Convert TRT outputs to action probabilities

    Args:
        engine_path: Path to the TensorRT engine.
        device_id: GPU device ID.
    """

    def __init__(
        self,
        engine_path: str | Path,
        device_id: int = 0,
    ):
        self.engine = TRTEngine(engine_path, device_id)
        logger.info("TRTInferencer initialized")

    def predict(
        self,
        image_features: np.ndarray,
        action_history: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run inference to get action probabilities and value estimate.

        Args:
            image_features: State features, shape (seq_len, feature_dim).
            action_history: Action tokens, shape (seq_len,).

        Returns:
            Tuple of (action_probs, value):
            - action_probs: (vocab_size,) probability distribution
            - value: scalar state value estimate
        """
        # Add batch dimension
        if image_features.ndim == 2:
            image_features = image_features[None, ...]  # (1, S, D)
        if action_history.ndim == 1:
            action_history = action_history[None, ...]  # (1, S)

        # Map to input names
        inputs = {}
        if "image_features" in self.engine.input_names:
            inputs["image_features"] = image_features.astype(np.float32)
        if "action_seq" in self.engine.input_names:
            inputs["action_seq"] = action_history.astype(np.int32)

        # Run inference
        outputs = self.engine.infer(inputs)

        # Extract outputs
        logits = outputs.get("action_logits", list(outputs.values())[0])
        values = outputs.get("state_values", list(outputs.values())[1])

        # Post-process: softmax for action probabilities
        logits_last = logits[0, -1, :]  # (vocab_size,)

        # Numerically stable softmax
        logits_last = logits_last - logits_last.max()
        exp_logits = np.exp(logits_last)
        action_probs = exp_logits / exp_logits.sum()

        value = float(values[0, -1, 0])

        return action_probs, value


class _Int8Calibrator:
    """
    INT8 calibrator for TensorRT.

    Feeds calibration data to TensorRT during engine building
    to determine optimal quantization parameters.
    """

    def __init__(
        self,
        calibration_data: np.ndarray,
        cache_file: Optional[str] = None,
    ):
        self.data = calibration_data
        self.cache_file = cache_file
        self.batch_idx = 0

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: List[str]) -> Optional[List[np.ndarray]]:
        if self.batch_idx >= len(self.data):
            return None

        batch = self.data[self.batch_idx : self.batch_idx + 1]
        self.batch_idx += 1
        return [batch]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)
