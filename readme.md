# GameRL - 用现代强化学习训练 AI 玩游戏

> 基于 [FengQuanLi/WZCQ](https://github.com/FengQuanLi/WZCQ) 原项目，使用最新技术栈全面重构。
> 支持王者荣耀、和平精英、原神等多游戏，通过 GameProfile 抽象层一键切换。

## 项目简介

WZCQ（王者荣耀）原项目创建于 2021 年，使用 PyTorch 1.9 + ResNet101 + 手写 Transformer + 不完整的 PPO 实现来训练 AI 玩王者荣耀。本项目在保留原始架构思路的基础上，使用 2025-2026 年的最新技术进行了全面现代化改造，并扩展为支持多游戏的通用 RL 游戏框架，更名为 **GameRL**。

## 原项目 vs 现代化版本

| 维度 | 原项目 (2021) | 现代化版本 (2026) |
|------|--------------|-------------------|
| **PyTorch** | 1.9.0 + CUDA 10.2 | 2.x + CUDA 12 |
| **特征提取** | ResNet101 (44M 参数) | ConvNeXt-Tiny (28M, 快 2x) |
| **视觉感知** | 原始 CNN 特征 (黑盒) | YOLO 目标检测 + 结构化状态向量 |
| **Transformer** | 手写 attention/embedding | `torch.nn.TransformerEncoder` (Pre-LN) + RoPE |
| **PPO** | 裁剪被注释掉，退化为 REINFORCE | 完整 PPO + GAE + 价值裁剪 + 熵正则 |
| **运行时** | 单循环 | 双循环调度器 (FastLoop 30-100Hz + SlowLoop 1Hz) |
| **战略层** | 无 | SlowLoop + VLM/LLM 战略推理回调 |
| **部署** | 无 | ONNX 导出 + TensorRT FP16/INT8 加速 |
| **多游戏** | 仅王者荣耀 | GameProfile 抽象层，3 游戏内置 |
| **触控方案** | pyminitouch (不支持 Android 10+) | ADB shell input (全版本兼容) |
| **截图** | win32gui 硬编码 (仅 Windows) | mss/win32/PyQt5 三后端可配置 |
| **代码结构** | 扁平文件 + 中文命名 | 模块化 Python 包 + 类型注解 |
| **配置** | 散落在各文件中的硬编码 | YAML + dataclass |
| **日志** | `print()` | Python logging + TensorBoard |
| **奖励系统** | 固定 7 字段 MOBA 专用 | 按游戏差异化的事件驱动 RewardShaper |
| **测试** | 无 | 126 个 pytest 单元测试全部通过 |
| **依赖管理** | conda environment.yml | pyproject.toml (PEP 621) |

## 项目结构

```
GameRL/
├── gamerl/                               # 主包
│   ├── config.py                       # YAML + dataclass 配置系统
│   │
│   ├── models/                         # 神经网络模型
│   │   ├── backbone.py                 # ConvNeXt-Tiny / EfficientNet / ResNet
│   │   ├── transformer.py              # Pre-LN Transformer + RoPE 策略网络
│   │   └── state_judgment.py           # 游戏事件分类模型
│   │
│   ├── agent/                          # 强化学习智能体
│   │   ├── ppo.py                      # 完整 PPO (裁剪 + GAE + 价值损失 + AMP)
│   │   └── memory.py                   # 高效 rollout 内存管理
│   │
│   ├── vision/                         # 视觉感知层 (NEW)
│   │   ├── detector.py                 # YOLO 目标检测 + MockDetector
│   │   ├── state_builder.py            # 检测结果 → 结构化状态向量
│   │   └── preprocess.py               # OpenCV 预处理 (resize/normalize/ROI)
│   │
│   ├── runtime/                        # 运行时调度 (NEW)
│   │   ├── scheduler.py                # 双循环: FastLoop (操作) + SlowLoop (战略)
│   │   └── latency_monitor.py          # 延迟监控 (p50/p95/p99 + 告警)
│   │
│   ├── inference/                      # 推理部署 (NEW)
│   │   ├── export_onnx.py              # PyTorch → ONNX 导出
│   │   └── trt_engine.py               # ONNX → TensorRT engine (FP16/INT8)
│   │
│   ├── profiles/                       # 游戏配置抽象层 (NEW)
│   │   ├── base.py                     # GameProfile 抽象基类
│   │   ├── honor_of_kings.py           # 王者荣耀 (130 vocab)
│   │   ├── peacekeeper.py              # 和平精英 (80 vocab, FPS 横屏)
│   │   ├── genshin.py                  # 原神 (72 vocab, 开放世界)
│   │   └── __init__.py                 # 注册表 + get_profile() 工厂
│   │
│   ├── environment/                    # 环境交互
│   │   ├── capture.py                  # 跨平台截图 (mss/win32/pyqt5)
│   │   ├── device.py                   # ADB 设备控制 + 动作映射
│   │   ├── game_env.py                 # 游戏环境封装 (接入 RewardShaper)
│   │   └── reward.py                   # 事件驱动奖励计算器 (NEW)
│   │
│   ├── data/                           # 数据处理
│   │   ├── collector.py                # 训练数据采集
│   │   ├── preprocessor.py             # 批量数据预处理
│   │   └── dataset.py                  # PyTorch Dataset + 序列分块
│   │
│   ├── training/                       # 训练管线
│   │   ├── trainer.py                  # 策略网络训练 (监督 + PPO)
│   │   └── state_trainer.py            # 状态判断模型训练
│   │
│   ├── utils/                          # 工具
│   │   ├── actions.py                  # 动作空间 (通用字符串列表)
│   │   ├── logging.py                  # 日志 + TensorBoard
│   │   └── masks.py                    # 注意力掩码生成
│   │
│   └── scripts/                        # CLI 入口脚本
│       ├── collect_data.py             # 数据采集
│       ├── preprocess.py               # 数据预处理
│       ├── train.py                    # 训练 (监督/PPO)
│       ├── train_state_model.py        # 状态模型训练
│       └── play.py                     # 运行 AI 玩游戏
│
├── configs/
│   └── default.yaml                    # 默认配置 (含 vision/runtime/inference 段)
├── tests/                              # 126 个单元测试
│   ├── test_models.py                  # 模型测试
│   ├── test_agent.py                   # PPO Agent 测试
│   ├── test_profiles.py                # GameProfile 测试
│   ├── test_vision.py                  # 视觉检测测试
│   ├── test_runtime.py                 # 运行时调度测试
│   └── test_reward.py                  # 奖励系统测试 (NEW)
├── original/                           # 原始 WZCQ 项目 (2021, 仅供参考)
│   ├── 模型_策略梯度.py                # 原始策略梯度模型
│   ├── 取训练数据.py                   # 原始数据采集
│   ├── 状态标注.py                     # 原始状态标注
│   ├── Batch.py / Embed.py / ...       # 原始 Transformer 组件
│   └── json/                           # 原始动作词汇表
├── weights/                            # 模型权重
├── scrcpy-win64-v4.1/                  # scrcpy + adb (已集成)
├── pyproject.toml                      # 依赖管理 (PEP 621)
└── README.md
```

## 安装

### 前置条件

1. **Python 3.10+**
2. **CUDA 12.x** (如需 GPU 训练；CPU 模式可跑全部测试)
3. **ADB** (Android Debug Bridge) - 用于手机控制
4. **scrcpy** - 用于低延迟屏幕镜像
5. 一台开启 USB 调试的安卓手机

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/FengQuanLi/WZCQ.git
cd GameRL

# 安装依赖 (推荐使用 venv)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -e ".[windows,capture,dev]"
```

### 配置 ADB 和 scrcpy

```bash
# 确认手机已连接
adb devices

# 启动 scrcpy 镜像 (低延迟视频流，用于实时画面捕获)
scrcpy --max-size 960
```

> **为什么需要 scrcpy？** `adb screencap` 单帧截图耗时 500ms-2s，而 scrcpy 走 H.264 视频流，取帧只要 16-33ms。scrcpy 负责看（视频流取帧），ADB 负责操作（点击/滑动），两者配合实现低延迟交互。

## 多游戏支持

通过 `GameProfile` 抽象层，通用 RL 基础设施（PPO、Transformer、训练管线）与游戏专属配置完全解耦。切换游戏只需改一行 YAML：

```yaml
game:
  name: "peacekeeper"  # 王者荣耀: honor_of_kings | 和平精英: peacekeeper | 原神: genshin
```

### 内置游戏 Profile

| 游戏 | Vocab | 分辨率 | 检测目标 | 奖励事件 | 终止事件 |
|------|-------|--------|---------|---------|---------|
| **王者荣耀** | 130 (10×13) | 2400×1080 | 英雄/血条/技能/塔/小兵 | kill_minion, kill_tower, kill_hero, assist_kill, attacked_by_tower, killed, death | death |
| **和平精英** | 80 (8×10) | 2340×1080 | 玩家/武器/载具/空投/安全区 | kill_enemy, down_enemy, got_killed, teammate_died, survived_frame, reached_final_circle, won_match, loot_item | got_killed, won_match |
| **原神** | 72 (8×9) | 2560×1440 | 角色/敌人/宝箱/NPC/采集物 | defeat_enemy, defeat_boss, character_downed, party_wiped, chest_opened, material_collected, quest_completed, exploration | party_wiped |

三个游戏的奖励事件**零重叠**——每个游戏定义完全独立的奖励语义，不再共用 MOBA 专用字段。

### 扩展新游戏

继承 `GameProfile`，实现抽象属性即可：

```python
from gamerl.profiles.base import GameProfile

class MyGameProfile(GameProfile):
    @property
    def name(self) -> str: return "my_game"
    @property
    def movements(self) -> list[str]: return ["up", "down", ...]
    @property
    def actions(self) -> list[str]: return ["attack", "skill", ...]
    @property
    def detection_classes(self) -> list[str]: return ["enemy", "item", ...]
    @property
    def reward_events(self) -> dict[str, float]:
        return {"kill_enemy": 5.0, "got_hit": -0.5, "normal": 0.01}
    @property
    def terminal_events(self) -> list[str]:
        return ["game_over"]
    # ... 其他属性
```

通用层（PPO、Transformer、训练管线、RewardShaper）完全不用改。

## 使用方法

### 1. 配置

编辑 `configs/default.yaml`：

```yaml
game:
  name: "honor_of_kings"       # 选择游戏

device:
  serial: ""                    # adb devices 查看的设备序列号
  window_title: "scrcpy"        # scrcpy 窗口标题
  capture_method: "mss"         # 截图方式
  screenshot_size: [960, 540]   # 截图尺寸

vision:
  backend: "yolo"               # 检测后端: yolo | mock
  model_path: "weights/yolo.pt" # YOLO 权重
  confidence: 0.5               # 置信度阈值
  max_objects: 50               # 最大检测对象数

runtime:
  fast_loop_hz: 30              # 快循环频率 (操作层)
  slow_loop_hz: 1               # 慢循环频率 (战略层)
  enable_latency_monitor: true  # 延迟监控

rewards:
  events:                       # 覆盖 Profile 默认奖励权重 (可选)
    kill_hero: 8.0
    killed: -3.0

inference:
  backend: "pytorch"            # pytorch | onnx | tensorrt
  precision: "fp16"             # fp32 | fp16 | int8
```

### 2. 采集训练数据

```bash
# AI 驱动采集 (需要已有模型)
python -m gamerl.scripts.collect_data --weights weights/policy_latest.pt

# 人工辅助采集 (半自动，推荐初始训练用)
python -m gamerl.scripts.collect_data --manual
```

### 3. 预处理数据

```bash
python -m gamerl.scripts.preprocess --data ../training_data
```

### 4. 训练状态判断模型

```bash
python -m gamerl.scripts.train_state_model --data ../labeled_data
```

### 5. 训练策略网络

```bash
# 监督预训练 (从演示数据学习)
python -m gamerl.scripts.train --mode supervised --data ../training_data

# PPO 强化学习微调
python -m gamerl.scripts.train --mode ppo \
    --weights weights/policy_latest.pt \
    --state-model weights/state_model_latest.pt
```

### 6. 运行 AI 玩游戏

```bash
python -m gamerl.scripts.play --weights weights/policy_latest.pt
```

### 7. 导出模型部署 (NEW)

```bash
# 导出 ONNX
python -c "
from gamerl.inference.export_onnx import export_policy_to_onnx
export_policy_to_onnx('weights/policy_latest.pt', 'weights/policy.onnx')
"

# 构建 TensorRT engine (需要 GPU + TensorRT)
python -c "
from gamerl.inference.trt_engine import TRTEngineBuilder
builder = TRTEngineBuilder(precision='fp16')
builder.build('weights/policy.onnx', 'weights/policy_trt.engine')
"
```

## 关键技术改进详解

### 1. 完整的 PPO 算法

原项目将 PPO 的核心——概率裁剪（clipping）——整段注释掉了，退化为最基础的 REINFORCE。现代化版本实现了标准 PPO：

```python
# PPO 裁剪目标函数
ratio = torch.exp(new_log_probs - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

# 价值函数裁剪
value_pred_clipped = old_values + torch.clamp(
    new_values - old_values, -clip_ratio, clip_ratio
)
value_loss = 0.5 * torch.max(
    (new_values - returns) ** 2,
    (value_pred_clipped - returns) ** 2
).mean()

# 熵正则化
entropy = dist.entropy().mean()
total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

### 2. YOLO 目标检测 + 结构化状态向量 (NEW)

原项目把整张图过 CNN 得到一个"黑盒"特征向量，模型只能从这个黑盒里学。现代化版本先用 YOLO 检测出结构化信息：

```python
# YOLO 检测 → 结构化状态
detections = detector.detect(frame)
# detections = [
#   Detection(class="enemy_hero", bbox=(0.45, 0.32, 0.08, 0.12), conf=0.92),
#   Detection(class="ally_tower", bbox=(0.50, 0.85, 0.06, 0.04), conf=0.88),
#   ...
# ]

state = state_builder.build(detections)
# state = StructuredState(
#   player_hp=0.75, enemy_count=3, nearest_enemy=(0.45, 0.32, 0.6),
#   skill_1_ready=True, skill_2_ready=False, tower_hp=0.80, ...
# )

# 转为固定维度向量送入 Transformer
state_vector = state.to_vector()  # (state_dim,)
```

模型决策更高效、更可解释、样本效率更高。支持 `ultralytics` YOLO 和 `MockDetector`（测试用）两种后端。

### 3. 双循环调度器 (NEW)

分层决策架构，快循环做操作，慢循环做战略：

```
┌──────────────────────────────────┐
│         SlowLoop (1Hz)           │  ← 战略层
│  分析全局局势 → 生成 StrategicDirective
│  (推塔/打团/撤退/farm)            │
│  支持 VLM/LLM 回调做高层推理      │
└──────────┬───────────────────────┘
           │ directive
           ▼
┌──────────────────────────────────┐
│        FastLoop (30-100Hz)       │  ← 操作层
│  截图 → YOLO检测 → 状态向量       │
│  → PPO决策 → ADB触控执行          │
└──────────────────────────────────┘
```

SlowLoop 不阻塞 FastLoop，通过 `StrategicDirective` 传递战略意图（如"优先推塔"会调整 FastLoop 的动作偏好）。

### 4. 事件驱动的差异化奖励系统 (NEW)

原项目使用固定 7 字段的 `RewardConfig`（`kill_minion_or_tower`, `kill_hero`, `attacked_by_tower`...），这些字段只适用于 MOBA。扩展到和平精英和原神时，大量字段填 `0.0`，语义不匹配。

现代化版本将奖励系统重构为**事件驱动**架构：

```python
# 每个 Profile 定义自己的奖励事件 (Dict[str, float])
# 王者荣耀 — MOBA 对抗
reward_events = {
    "kill_minion": 2.0,      # 补兵
    "kill_tower": 5.0,       # 推塔
    "kill_hero": 5.0,        # 击杀英雄
    "assist_kill": 2.0,      # 助攻
    "attacked_by_tower": -0.5, # 被塔攻击
    "killed": -2.0,          # 被击杀
    "death": -1.0,           # 阵亡
    "normal": 0.01,          # 存活基线
    "other": -0.003,         # 无意义操作惩罚
}
terminal_events = ["death"]

# 和平精英 — FPS 生存
reward_events = {
    "kill_enemy": 10.0,        # 击杀敌人
    "down_enemy": 3.0,         # 击倒敌人
    "got_killed": -5.0,        # 被击杀
    "teammate_died": -1.0,     # 队友阵亡
    "survived_frame": 0.01,    # 存活
    "reached_final_circle": 5.0, # 进决赛圈
    "won_match": 20.0,         # 吃鸡
    "loot_item": 0.5,          # 拾取物资
    "normal": 0.01, "other": -0.001,
}
terminal_events = ["got_killed", "won_match"]

# 原神 — 开放世界 RPG
reward_events = {
    "defeat_enemy": 1.0,       # 击败敌人
    "defeat_boss": 5.0,        # 击败 Boss
    "character_downed": -3.0,  # 角色倒下
    "party_wiped": -10.0,      # 全队阵亡
    "chest_opened": 2.0,       # 开宝箱
    "material_collected": 0.3, # 采集材料
    "quest_completed": 10.0,   # 完成任务
    "exploration": 0.005,      # 探索奖励
    "normal": 0.005, "other": -0.001,
}
terminal_events = ["party_wiped"]
```

**RewardShaper** (`gamerl/environment/reward.py`) 负责完整计算管线：

```
StateJudgmentModel / 自定义 callback → 事件分类 → 查 reward_events 权重
    → 叠加战略指令权重 (StrategicDirective) → 检测终止事件 → 标量 reward + done
```

- `from_profile(profile)` 一行创建
- 支持战略指令叠加（如"优先推塔"可临时提高 `kill_tower` 权重）
- 自动统计事件分布、累计奖励、平均奖励

`GameEnvironment.step()` 每步调用 `RewardShaper.compute_reward()`，返回真实 reward 和 done 标志，不再硬编码 `reward = 0.0`。

YAML 配置可覆盖 Profile 默认权重：

```yaml
rewards:
  events:
    kill_hero: 8.0      # 覆盖默认 5.0
    killed: -3.0        # 覆盖默认 -2.0
```

### 5. TensorRT 部署加速 (NEW)

部署链路：PyTorch (FP32, ~40ms) → ONNX (跨平台) → TensorRT (FP16 ~15ms / INT8 ~8ms)

```python
# ONNX 导出 (动态 batch/seq_len, onnx-simplifier 优化)
export_policy_to_onnx("policy.pt", "policy.onnx")

# TensorRT engine 构建 (FP16/INT8, 动态 batch profile)
builder = TRTEngineBuilder(precision="fp16")
builder.build("policy.onnx", "policy.engine")
```

### 6. RoPE 旋转位置编码

原项目使用学习式位置 embedding，现代化版本使用 RoPE：

- 更好的长度泛化能力
- 相对位置编码，更适合序列建模
- 无需额外参数

### 7. Pre-LayerNorm Transformer

原项目使用 Post-LN（先注意力/FFN 再归一化），现代化版本使用 Pre-LN：

- 训练更稳定，不需要 warmup
- 兼容 Flash Attention
- 使用 PyTorch 原生 `nn.TransformerEncoder`
- 注意：`norm_first=True` + 布尔掩码会导致 NaN，需转为 float mask (0/-inf)

### 8. 跨平台截图 + ADB 触控

- **截图三后端**: mss (跨平台最快) / win32 (Windows 原生) / PyQt5 (scrcpy 窗口捕获)
- **触控**: ADB shell input 替代 pyminitouch，全 Android 版本兼容

## 运行测试

```bash
# 全部 126 个测试
pytest tests/ -v

# 仅运行特定模块
pytest tests/test_vision.py -v      # 视觉检测
pytest tests/test_runtime.py -v     # 运行时调度
pytest tests/test_profiles.py -v    # 游戏 Profile
pytest tests/test_models.py -v      # 神经网络模型
pytest tests/test_agent.py -v       # PPO Agent
pytest tests/test_reward.py -v      # 奖励系统
```

## 查看训练日志

```bash
tensorboard --logdir runs/
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.x |
| 图像骨干 | ConvNeXt-Tiny / EfficientNet / ResNet |
| 目标检测 | YOLO (ultralytics) |
| 图像处理 | OpenCV (opencv-python) |
| 序列模型 | nn.TransformerEncoder + RoPE |
| 强化学习 | PPO + GAE |
| 奖励系统 | 事件驱动 RewardShaper (按游戏差异化) |
| 设备控制 | ADB shell input |
| 屏幕捕获 | scrcpy + mss/win32/PyQt5 |
| 部署加速 | ONNX + TensorRT (FP16/INT8) |
| 配置管理 | YAML + dataclass |
| 依赖管理 | pyproject.toml (PEP 621) |
| 测试 | pytest (126 tests) |
| 日志 | Python logging + TensorBoard |

## 致谢

- 原项目作者: [FengQuanLi](https://github.com/FengQuanLi)
- ResNet, Transformer 等基础架构来自 PyTorch/torchvision
- PPO 算法参考 [PPO 论文](https://arxiv.org/abs/1707.06347) 和 [OpenAI Spinning Up](https://spinningup.openai.com/)
- YOLO 目标检测来自 [Ultralytics](https://github.com/ultralytics/ultralytics)
- scrcpy 来自 [Genymobile](https://github.com/Genymobile/scrcpy)

## License

MIT License (见 [LICENSE](LICENSE))
