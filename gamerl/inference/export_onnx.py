"""
Export PyTorch models to ONNX format.

Supports exporting both the policy network (TransformerPolicy) and the
detection model (YOLO) to ONNX for cross-platform deployment and
TensorRT optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("gamerl.inference")


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""

    opset_version: int = 17
    dynamic_batch: bool = True
    dynamic_seq_len: bool = True
    simplify: bool = True  # Run onnx-simplifier
    input_names: tuple = ("image_features", "action_seq")
    output_names: tuple = ("action_logits", "state_values")


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: Tuple[int, ...] = (1, 300, 768),
    action_shape: Tuple[int, ...] = (1, 300),
    config: Optional[ExportConfig] = None,
    device: str = "cpu",
    vocab_size: Optional[int] = None,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export.
        output_path: Path to save the ONNX file.
        input_shape: Shape of the image features input (batch, seq_len, feature_dim).
        action_shape: Shape of the action sequence input (batch, seq_len).
        config: Export configuration. Uses defaults if None.
        device: Device to run export on.
        vocab_size: Vocabulary size for dummy action tokens.  If None,
            inferred from ``model.action_embed.num_embeddings``.

    Returns:
        Path to the saved ONNX file.
    """
    if config is None:
        config = ExportConfig()

    # Infer vocab size from the model's embedding layer if not given
    if vocab_size is None:
        if hasattr(model, "action_embed") and hasattr(model.action_embed, "num_embeddings"):
            vocab_size = model.action_embed.num_embeddings
        else:
            vocab_size = 7  # Universal action space default

    model = model.to(device)
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy inputs
    dummy_features = torch.randn(*input_shape, device=device)
    dummy_actions = torch.randint(0, vocab_size, action_shape, device=device)

    # Set dynamic axes if configured
    dynamic_axes = None
    if config.dynamic_batch or config.dynamic_seq_len:
        dynamic_axes = {}
        if config.dynamic_batch:
            dynamic_axes[config.input_names[0]] = {0: "batch"}
            dynamic_axes[config.input_names[1]] = {0: "batch"}
            dynamic_axes[config.output_names[0]] = {0: "batch"}
            dynamic_axes[config.output_names[1]] = {0: "batch"}
        if config.dynamic_seq_len:
            dynamic_axes[config.input_names[0]][1] = "seq_len"
            dynamic_axes[config.input_names[1]][1] = "seq_len"
            dynamic_axes[config.output_names[0]][1] = "seq_len"
            dynamic_axes[config.output_names[1]][1] = "seq_len"

    logger.info(f"Exporting model to ONNX: {output_path}")
    logger.info(f"  Input shapes: features={input_shape}, actions={action_shape}")
    logger.info(f"  Dynamic axes: {dynamic_axes}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_features, dummy_actions),
            str(output_path),
            export_params=True,
            opset_version=config.opset_version,
            do_constant_folding=True,
            input_names=list(config.input_names),
            output_names=list(config.output_names),
            dynamic_axes=dynamic_axes,
        )

    logger.info(f"ONNX export complete: {output_path}")

    # Optionally simplify
    if config.simplify:
        _simplify_onnx(output_path)

    return output_path


def _simplify_onnx(onnx_path: Path) -> None:
    """Run onnx-simplifier to reduce model complexity."""
    try:
        import onnx
        from onnxsim import simplify

        logger.info("Simplifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        simplified, check = simplify(onnx_model)

        if check:
            onnx.save(simplified, str(onnx_path))
            logger.info("ONNX simplification successful")
        else:
            logger.warning("ONNX simplification check failed, keeping original")
    except ImportError:
        logger.warning(
            "onnx-simplifier not installed. Install with: pip install onnx onnxsim"
        )
    except Exception as e:
        logger.warning(f"ONNX simplification failed: {e}")


def export_yolo_to_onnx(
    model_path: str,
    output_path: str | Path,
    imgsz: int = 640,
    simplify: bool = True,
) -> Path:
    """
    Export a YOLO model to ONNX format.

    Args:
        model_path: Path to YOLO .pt weights.
        output_path: Path to save the ONNX file.
        imgsz: Input image size.
        simplify: Whether to simplify the model.

    Returns:
        Path to the saved ONNX file.
    """
    output_path = Path(output_path)

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            dynamic=True,
            opset=17,
        )

        # ultralytics saves to the same directory as the model
        auto_path = Path(model_path).with_suffix(".onnx")
        if auto_path != output_path and auto_path.exists():
            auto_path.rename(output_path)

        logger.info(f"YOLO exported to ONNX: {output_path}")
        return output_path

    except ImportError:
        logger.error("ultralytics not installed for YOLO export")
        raise


def verify_onnx(
    onnx_path: str | Path,
    test_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
    vocab_size: Optional[int] = None,
) -> bool:
    """
    Verify an ONNX model by comparing outputs with PyTorch.

    Args:
        onnx_path: Path to the ONNX file.
        test_inputs: Optional test inputs. Uses random if None.
        vocab_size: Vocabulary size for random test actions (default 7).

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.error("onnx and onnxruntime required for verification")
        return False

    onnx_path = Path(onnx_path)

    # Load and check
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")

    # Create session
    session = ort.InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in session.get_inputs()]

    # Generate test inputs if not provided
    if test_inputs is None:
        vs = vocab_size or 7
        test_inputs = (
            torch.randn(1, 10, 768),
            torch.randint(0, vs, (1, 10)),
        )

    # Run inference
    feed = {
        name: inp.numpy()
        for name, inp in zip(input_names, test_inputs)
    }
    outputs = session.run(None, feed)

    logger.info(f"ONNX inference successful. Output shapes: {[o.shape for o in outputs]}")
    return True
