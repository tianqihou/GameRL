"""
Configuration management using dataclasses and YAML.

Replaces the scattered hardcoded constants in the original project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class DeviceConfig:
    """Android device and screen capture configuration."""

    serial: str = ""
    window_title: str = "scrcpy"
    capture_method: str = "mss"  # "win32" | "mss" | "pyqt5"
    screenshot_size: tuple[int, int] = (960, 540)
    crop_box: Optional[tuple[int, int, int, int]] = None


@dataclass
class ModelConfig:
    """Neural network model architecture."""

    backbone: str = "convnext_tiny"  # "convnext_tiny" | "efficientnet_v2_s" | "resnet50" | "resnet101"
    backbone_grid_size: int = 6
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0
    max_seq_len: int = 300
    pretrained: bool = True
    backbone_half: bool = True


@dataclass
class AgentConfig:
    """PPO agent hyperparameters."""

    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-5
    ppo_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    minibatch_size: int = 16


@dataclass
class GameConfig:
    """Game profile selection."""

    name: str = "honor_of_kings"  # "honor_of_kings" | "peacekeeper" | "genshin"
    resolution: Optional[tuple[int, int]] = None  # Override profile default


@dataclass
class StateModelConfig:
    """State judgment (event classification) model."""

    d_model: int = 768
    n_layers: int = 2
    n_heads: int = 12
    dropout: float = 0.0
    num_classes: int = 6
    # Vocabulary size for action embedding.  None = auto-detect from the
    # game profile (7 for universal action space, per-game size for legacy).
    vocab_size: Optional[int] = None


@dataclass
class RewardConfig:
    """
    Flexible reward configuration as a key-value map.

    Each game profile defines its own reward event names and default
    weights. The YAML 'rewards' section can override any of them.

    Common keys across all games:
    - 'normal': per-frame baseline reward (encourage survival / exploration)
    - 'other': fallback for unclassified events

    Game-specific keys are defined in each GameProfile's reward_events property.
    """

    events: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.01,
        "other": -0.003,
    })

    # Reward clipping bounds (min, max) for PPO stability.
    # Extreme reward spikes destabilize the value function.
    clip_min: float = -10.0
    clip_max: float = 10.0

    def get(self, event: str, default: float = 0.0) -> float:
        """Get reward weight for an event."""
        return self.events.get(event, default)

    def update(self, overrides: Dict[str, float]) -> None:
        """Merge override weights into the current config."""
        self.events.update(overrides)


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    epochs: int = 100
    chunk_size: int = 600
    stride: int = 600
    use_amp: bool = True
    save_every: int = 1
    resume: Optional[str] = None
    log_dir: str = "runs"
    weights_dir: str = "weights"


@dataclass
class CollectionConfig:
    """Data collection (gameplay) configuration."""

    save_dir: str = "../training_data"
    learn_every: int = 15000
    target_fps: int = 5
    auto_buy_interval: int = 50


@dataclass
class PathsConfig:
    """File paths for vocabularies and mappings."""

    action_vocab: str = "json/action_vocab.json"
    action_commands: str = "json/action_commands.json"


@dataclass
class VisionConfig:
    """Computer vision pipeline configuration."""

    enabled: bool = True
    detector_backend: str = "mock"  # "yolo" | "mock"
    model_path: str = ""  # Path to YOLO weights (empty for mock)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_size: tuple[int, int] = (640, 640)
    max_enemies: int = 5
    max_towers: int = 6
    max_minions: int = 10


@dataclass
class RuntimeConfig:
    """Dual-loop scheduler configuration."""

    fast_loop_hz: float = 30.0
    slow_loop_hz: float = 1.0
    enable_latency_monitor: bool = True
    latency_alert_threshold_ms: float = 50.0


@dataclass
class InferenceConfig:
    """Inference optimization configuration."""

    backend: str = "pytorch"  # "pytorch" | "onnx" | "tensorrt"
    onnx_path: str = "weights/policy.onnx"
    engine_path: str = "weights/policy.engine"
    fp16: bool = True
    int8: bool = False
    device_id: int = 0


@dataclass
class ImitationConfig:
    """Behavior cloning (imitation learning) configuration.

    The recommended training workflow is:
    1. Collect human demonstrations (collect_data --manual)
    2. Behavior cloning pretraining (train --mode supervised)
    3. PPO fine-tuning (train --mode ppo)
    """

    enabled: bool = True
    # Number of BC pretraining epochs before PPO
    bc_epochs: int = 20
    # Directory containing demonstration data
    dataset_path: str = "data/demonstrations"


@dataclass
class StrategicRewardsConfig:
    """Long-horizon reward components layered on top of event rewards.

    These weights are added to the event-driven reward when the state
    pipeline can supply the corresponding signals.  All default to 0.0
    (disabled) so existing behavior is unchanged unless configured.
    """

    # Progress toward strategic objectives (tower damage, objective control)
    objective_progress: float = 0.0
    # Per-frame survival bonus (encourages staying alive)
    survival: float = 0.0
    # Exploration bonus for visiting new screen regions
    exploration: float = 0.0


@dataclass
class Config:
    """Top-level configuration container."""

    game: GameConfig = field(default_factory=GameConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    state_model: StateModelConfig = field(default_factory=StateModelConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    strategic_rewards: StrategicRewardsConfig = field(default_factory=StrategicRewardsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        def _build_section(dataclass_type, section_data):
            if section_data is None:
                return dataclass_type()
            # Filter to only known fields
            valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
            filtered = {k: v for k, v in section_data.items() if k in valid_fields}
            return dataclass_type(**filtered)

        def _build_rewards(section_data):
            """Build RewardConfig from YAML rewards section.

            YAML format is a flat key-value map of event_name -> weight.
            These override the profile's default reward_events.
            """
            if section_data is None:
                return RewardConfig()
            return RewardConfig(events=dict(section_data))

        return cls(
            game=_build_section(GameConfig, data.get("game")),
            device=_build_section(DeviceConfig, data.get("device")),
            model=_build_section(ModelConfig, data.get("model")),
            agent=_build_section(AgentConfig, data.get("agent")),
            state_model=_build_section(StateModelConfig, data.get("state_model")),
            rewards=_build_rewards(data.get("rewards")),
            training=_build_section(TrainingConfig, data.get("training")),
            collection=_build_section(CollectionConfig, data.get("collection")),
            paths=_build_section(PathsConfig, data.get("paths")),
            vision=_build_section(VisionConfig, data.get("vision")),
            runtime=_build_section(RuntimeConfig, data.get("runtime")),
            inference=_build_section(InferenceConfig, data.get("inference")),
            imitation=_build_section(ImitationConfig, data.get("imitation")),
            strategic_rewards=_build_section(
                StrategicRewardsConfig, data.get("strategic_rewards")
            ),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {}
        for field_name in self.__dataclass_fields__:
            section = getattr(self, field_name)
            if isinstance(section, RewardConfig):
                data[field_name] = dict(section.events)
            else:
                data[field_name] = {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in section.__dict__.items()
                }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
