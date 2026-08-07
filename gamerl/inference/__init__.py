"""
Inference optimization and deployment.

Provides ONNX export and TensorRT engine building for production deployment:

    PyTorch (.pt) → ONNX (.onnx) → TensorRT (.engine)

Supports FP16 and INT8 precision for reduced latency.

Usage:
    from gamerl.inference import export_to_onnx, build_trt_engine

    # Export model to ONNX
    export_to_onnx(policy, "weights/policy.onnx", input_shape=(1, 300, 768))

    # Build TensorRT engine (FP16)
    build_trt_engine("weights/policy.onnx", "weights/policy.engine", fp16=True)
"""

from .export_onnx import export_to_onnx, ExportConfig
from .trt_engine import (
    TRTEngine,
    build_trt_engine,
    TRTConfig,
    TRTInferencer,
)

__all__ = [
    "export_to_onnx",
    "ExportConfig",
    "TRTEngine",
    "build_trt_engine",
    "TRTConfig",
    "TRTInferencer",
]
