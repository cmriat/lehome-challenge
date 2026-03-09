# RLinf 集成指南：用 RL 提升 Pi0.5 叠衣性能

> **文档状态**: 渐进式实践文档，随实际集成进度持续更新。
> **创建时间**: 2026-03-09
> **环境管理**: pixi (本项目使用 pixi 管理依赖，非 Docker)

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [实践策略](#2-实践策略)
3. [RLinf 框架概览](#3-rlinf-框架概览)
4. [Phase 1: 安装与环境搭建](#4-phase-1-安装与环境搭建)
5. [Phase 2: 环境 Wrapper 适配](#5-phase-2-环境-wrapper-适配)
6. [Phase 3: Pi0.5 + PPO 单类型验证](#6-phase-3-pi05--ppo-单类型验证)
7. [Phase 4: 扩展到全类型](#7-phase-4-扩展到全类型)
8. [参考配置与超参](#8-参考配置与超参)
9. [常见问题与排查](#9-常见问题与排查)
10. [参考资料](#10-参考资料)

---

## 1. 背景与动机

### 当前 Pi0.5 评估结果 (step 124k)

| 类别 | Overall 成功率 | Seen | Unseen |
|------|---------------|------|--------|
| Pant Short | 88.33% | 94% | 60% |
| Top Long | 75.00% | 76% | 70% |
| Pant Long | 66.67% | 68% | 60% |
| Top Short | 45.00% | 50% | 20% |

### 为什么需要 RL

- 纯 SFT (行为克隆) 从 80k -> 124k step 仅提升 ~1.7%，接近瓶颈
- Top Short 类别成功率仅 45%，需要新的优化范式
- Unseen 泛化不稳定，RL 的在线探索能力有望改善

### 为什么选 RLinf 而非自建 RWR

| 维度 | 自建 RWR | RLinf |
|------|---------|-------|
| 本质 | 加权行为克隆 (无探索) | 真正的 RL (PPO/SAC/GRPO) |
| Flow Matching 适配 | 未处理去噪过程 | Flow-SDE / Flow-Noise 专门设计 |
| Pi0.5 支持 | 手动对接 | 原生支持 + LoRA |
| 并行收集 | 单环境串行 | 多 GPU 分布式 |
| 已验证效果 | 未知 | Pi0.5 从 40% -> 84.8% (LIBERO) |

---

## 2. 实践策略

### 渐进式集成原则

本文档采用**分阶段、边做边验证**的策略，而非一步到位：

```
Phase 1: 安装搭建     -> 目标: rlinf 可 import, 基础 demo 跑通
Phase 2: 环境 Wrapper  -> 目标: LeHome 衣物环境接入 RLinf
Phase 3: 单类型验证    -> 目标: Top Short 上跑通 PPO, 观察是否有提升
Phase 4: 全类型扩展    -> 目标: 四类衣物全面 RL fine-tune
```

### 每个 Phase 的工作方式

1. **先跑通最小可行路径** — 不追求最优，先确认可行
2. **遇到问题记录到本文档** — 在对应 Phase 下维护 "已知问题" 列表
3. **每个 Phase 结束后做评估** — 对比 SFT baseline，确认是否继续

### 预估时间

| Phase | 预估时间 | 关键风险 |
|-------|---------|---------|
| Phase 1 | 1-2 天 | pixi 与 RLinf 依赖冲突 |
| Phase 2 | 3-5 天 | Isaac Lab 衣物仿真 + RLinf 接口对接 |
| Phase 3 | 1-2 周 | 训练不稳定、GPU 资源、布料粒子崩溃 |
| Phase 4 | 1-2 周 | 多类型超参调优 |

---

## 3. RLinf 框架概览

### 架构

```
RLinf
├── rlinf/
│   ├── envs/           # 环境适配层 (isaaclab/, libero/, maniskill/, ...)
│   ├── models/         # 模型封装 (openpi/pi0.5, openvla, gr00t, ...)
│   ├── algorithms/     # RL 算法 (PPO, GRPO, SAC, ...)
│   ├── runners/        # 训练/评估 runner
│   ├── workers/        # 分布式 worker (actor, rollout, reward, ...)
│   └── hybrid_engines/ # 训练后端 (FSDP, Megatron, ...)
├── examples/
│   └── embodiment/
│       ├── config/     # Hydra 配置 (含 Pi0.5 + Isaac Lab 示例)
│       └── train_embodied_agent.py  # 训练入口
└── requirements/
    └── install.sh      # 安装脚本
```

### 关键概念

- **Actor**: 模型训练组件，负责梯度更新
- **Rollout**: 模型推理组件，负责 action 采样 (可与 Actor 分离到不同 GPU)
- **Env**: 环境组件，负责仿真和 reward 计算
- **FSDP**: 训练后端，支持模型并行和混合精度

### Pi0.5 在 RLinf 中的模型配置

```yaml
# examples/embodiment/config/model/pi0_5.yaml
model_type: "openpi"
num_action_chunks: 10    # action chunk 大小
action_dim: 7            # 单臂 7, 双臂需改为 12
is_lora: False           # 是否使用 LoRA
lora_rank: 32
num_steps: 5             # 去噪步数
openpi:
  config_name: "pi05_libero"
  train_expert_only: True   # 默认只训练 action expert (冻结 VLM)
  noise_method: "flow_sde"  # RL 算法: flow_sde 或 flow_noise
```

---

## 4. Phase 1: 安装与环境搭建

### 4.1 安装策略

由于本项目使用 **pixi** 管理环境，而 RLinf 默认使用 Docker 或 uv venv，
我们采用 **pip 库模式** 将 RLinf 核心安装到 pixi 环境中：

```bash
# 在 pixi 环境中安装 RLinf 核心库
pixi run pip install rlinf[embodied]
```

> **注意**: `rlinf` pip 包只包含核心调度/训练逻辑，不包含环境和模型依赖。
> 环境 (Isaac Lab) 和模型 (lerobot/Pi0.5) 依赖由我们的 pixi.toml 已有管理。

### 4.2 克隆 RLinf 仓库 (获取配置和示例)

```bash
# 在项目外部克隆，不污染主项目
cd /home/maozan/code
git clone https://github.com/RLinf/RLinf.git
cd RLinf
```

我们需要的主要是:
- `examples/embodiment/config/` — 参考配置模板
- `rlinf/envs/isaaclab/` — Isaac Lab 环境 wrapper 参考实现

### 4.3 验证安装

```bash
pixi shell
python -c "import rlinf; print('RLinf version:', rlinf.__version__)"
```

### 4.4 可能的依赖冲突

| 依赖 | pixi 当前版本 | RLinf 要求 | 处理方式 |
|------|-------------|-----------|---------|
| torch | 2.7.0 | >= 2.4.0 | 兼容 |
| gymnasium | >= 0.29.1 | >= 0.29 | 兼容 |
| lerobot | 0.4.3 | 自带 openpi wrapper | 可能冲突，需确认 |
| ray | 未安装 | 需要 (分布式调度) | 需追加安装 |

```bash
# 追加安装 ray (RLinf 的分布式调度依赖)
pixi run pip install "ray[default]>=2.30"
```

### Phase 1 验收标准

- [ ] `import rlinf` 成功
- [ ] `import ray` 成功
- [ ] 能加载 RLinf 的 Hydra 配置文件
- [ ] pixi 环境中原有的 eval 脚本仍能正常运行 (回归测试)

### Phase 1 已知问题

<!-- 在实际操作过程中记录遇到的问题和解决方案 -->

---

## 5. Phase 2: 环境 Wrapper 适配

这是集成的**核心难点**。需要将 LeHome 的衣物折叠环境包装为 RLinf 可识别的格式。

### 5.1 RLinf 环境接口要求

参考 `rlinf/envs/isaaclab/tasks/stack_cube.py`，需要实现:

```python
class IsaaclabGarmentEnv(IsaaclabBaseEnv):
    """LeHome 衣物折叠环境的 RLinf Wrapper"""

    def _make_env_function(self):
        """返回一个创建 Isaac Lab 环境的函数"""
        ...

    def _wrap_obs(self, obs):
        """将环境原始 obs 转换为 RLinf 标准格式"""
        ...
```

### 5.2 LeHome 环境的观测空间

当前 LeHome 衣物环境 (`garment_bi_v2.py`) 的 observation 结构:

```python
{
    "observation.state": joint_pos,                    # (12,) 双臂关节角度
    "observation.images.top_rgb": top_camera_rgb,      # (H, W, 3) 顶部相机
    "observation.images.left_rgb": left_camera_rgb,    # (H, W, 3) 左侧相机
    "observation.images.right_rgb": right_camera_rgb,  # (H, W, 3) 右侧相机
}
```

### 5.3 观测映射

RLinf (openpi 模型) 期望的 obs 格式:

```python
{
    "main_images": ...,      # 主视角图像
    "wrist_images": ...,     # 腕部/辅助视角图像
    "states": ...,           # 机器人状态
    "task_descriptions": ..., # 任务描述文本
}
```

映射关系:

| RLinf 字段 | LeHome 字段 | 说明 |
|-----------|-------------|------|
| `main_images` | `observation.images.top_rgb` | 顶部俯视图 |
| `wrist_images` | `observation.images.left_rgb` 或 `right_rgb` | 需要确认哪个更有效 |
| `states` | `observation.state` | 12 维关节角 (双臂) |
| `task_descriptions` | `"fold the garment"` | 固定任务描述 |

### 5.4 Wrapper 骨架代码

在 `scripts/rl/` 下创建 wrapper:

```python
# scripts/rl/rlinf_env_wrapper.py

import gymnasium as gym
import torch
from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


class LeHomeGarmentEnv(IsaaclabBaseEnv):
    """RLinf wrapper for LeHome garment folding environment."""

    def _make_env_function(self):
        """Create the Isaac Lab garment environment."""
        cfg = self.cfg

        def make_env():
            import os
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher
            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            import lehome.tasks.bedroom  # noqa: register env
            from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg

            env_cfg = GarmentEnvCfg()
            env_cfg.garment_name = cfg.init_params.garment_name
            env_cfg.garment_version = "Release"
            env_cfg.scene.num_envs = cfg.init_params.num_envs

            env = gym.make(
                "LeHome-BiSO101-Direct-Garment-v2",
                cfg=env_cfg,
            ).unwrapped
            env.initialize_obs()
            return env, sim_app

        return make_env

    def _wrap_obs(self, obs):
        """Convert LeHome observations to RLinf format."""
        instruction = [self.task_description] * self.num_envs
        env_obs = {
            "main_images": obs["observation.images.top_rgb"],
            "wrist_images": obs["observation.images.left_rgb"],
            "states": obs["observation.state"],
            "task_descriptions": instruction,
        }
        return env_obs
```

### 5.5 衣物切换的特殊处理

LeHome 环境需要在不同衣物间切换评估。需要额外处理:

- **方案 A**: 每个 garment variant 作为独立的 task variant (推荐)
- **方案 B**: 在 env wrapper 中实现 garment 轮换
- **方案 C**: 训练时固定单个 garment，评估时切换

建议 Phase 3 先用**方案 C** (最简单)，Phase 4 再扩展到方案 A。

### 5.6 Reward 设计

当前环境的 reward 基于关键点距离 (如 `dist(p[0], p[4]) <= threshold`)。
RLinf 支持 `reward_type: chunk_level` (整个 action chunk 的累计 reward)。

需要确认:
- [ ] `env._get_rewards()` 的返回值是否适合直接作为 RL reward
- [ ] 是否需要 reward shaping (如成功 +1, 失败 0, 中间步骤用距离变化)

### Phase 2 验收标准

- [ ] `LeHomeGarmentEnv` 类可正常实例化
- [ ] `reset()` 返回正确格式的 obs
- [ ] `step(action)` 返回 (obs, reward, terminated, truncated, info)
- [ ] 手动测试: 随机 action 跑完一个 episode 不报错

### Phase 2 已知问题

<!-- 实际操作中记录 -->

---

## 6. Phase 3: Pi0.5 + PPO 单类型验证

### 6.1 选择 Top Short 作为验证目标

- 当前最弱类别 (45% overall, Seen 仅 50%)
- 提升空间最大，效果最容易观察
- 如果 RL 在这里有效，说明方法本身有价值

### 6.2 参考配置

基于 `libero_spatial_ppo_openpi_pi05.yaml` 适配:

```yaml
# configs/rl/rlinf_top_short_ppo_pi05.yaml (需创建)
defaults:
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config

runner:
  task_type: embodied
  max_epochs: 500
  save_interval: 50
  logger:
    experiment_name: "lehome_top_short_ppo_pi05"

algorithm:
  rollout_epoch: 4          # 每 epoch 收集的 rollout 轮数
  update_epoch: 1
  adv_type: gae
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  kl_beta: 0.0             # 无额外 KL 惩罚 (Pi0.5 已冻结 VLM)
  reward_type: chunk_level
  logprob_type: chunk_level
  sampling_params:
    temperature_train: 1.0
    temperature_eval: 0.6

env:
  train:
    total_num_envs: 8       # 受限于布料仿真的 GPU 开销，从小开始
    max_episode_steps: 600
    max_steps_per_rollout_epoch: 600

actor:
  micro_batch_size: 8
  global_batch_size: 64
  model:
    model_path: "/home/maozan/code/data/ckpts/pi05_step124k"
    action_dim: 12          # 双臂 12 维
    num_action_chunks: 5    # 需确认 (Pi0.5 的 chunk_size)
    num_steps: 3            # 去噪步数 (减少以加速)
    add_value_head: True
    openpi:
      train_expert_only: True    # 冻结 VLM, 只训练 action expert
      noise_method: "flow_sde"   # Flow-SDE 方法

  optim:
    lr: 5.0e-6
    value_lr: 1.0e-4
    clip_grad: 1.0
```

### 6.3 关键超参说明

| 参数 | 值 | 原因 |
|------|-----|------|
| `train_expert_only: True` | 冻结 VLM | GPU 显存有限 + 防止灾难性遗忘 |
| `noise_method: flow_sde` | Flow-SDE | 适合 flow matching policy 的 RL 算法 |
| `num_steps: 3` | 减少去噪步数 | 加速推理 (训练时需频繁采样) |
| `total_num_envs: 8` | 少量环境 | 布料仿真 GPU 开销大，先小规模验证 |
| `lr: 5e-6` | 小学习率 | 保护预训练知识 |

### 6.4 训练启动

```bash
# 需要根据实际适配情况调整
cd /home/maozan/code/RLinf

# 设置环境变量
export EMBODIED_PATH=$(pwd)/examples/embodiment

# 启动训练 (单机)
pixi run python examples/embodiment/train_embodied_agent.py \
    --config-path configs/rl \
    --config-name rlinf_top_short_ppo_pi05
```

### 6.5 评估对比

训练完成后，使用已有的 eval 脚本对比:

```bash
# 评估 RL fine-tuned 模型
bash scripts/eval_single_type.sh top_short \
    --policy_path outputs/rlinf_top_short/checkpoints/best_model
```

对比指标:
- Top Short Seen 成功率 (baseline: 50%)
- Top Short Unseen 成功率 (baseline: 20%)
- 其他类型的成功率 (检测灾难性遗忘)

### Phase 3 验收标准

- [ ] 训练可正常启动，loss 有下降
- [ ] 训练过程中 rollout 的成功率有上升趋势
- [ ] Top Short 评估成功率 > 50% (超过 SFT baseline)
- [ ] 其他类型未明显退化

### Phase 3 已知问题

<!-- 实际操作中记录 -->

---

## 7. Phase 4: 扩展到全类型

### 7.1 多任务 RL 训练

Phase 3 成功后，扩展到 4 种衣物类型:

- 方案 A: 4 种类型联合训练 (共享 action expert)
- 方案 B: 每种类型独立训练 (4 个模型)
- 方案 C: 先联合训练基础，再各自 fine-tune

### 7.2 利用 LoRA 避免遗忘

```yaml
actor:
  model:
    is_lora: True
    lora_rank: 32
```

LoRA 优势:
- 只更新少量参数，减少遗忘风险
- 可以为每种衣物类型训练独立的 LoRA adapter
- 推理时可合并回主模型

### 7.3 Unseen 泛化策略

- 训练时使用全部 Seen 衣物 (10 种 per type)
- 加入 domain randomization (初始位姿、光照等)
- 评估时关注 Unseen 成功率变化

### Phase 4 验收标准

- [ ] 4 种衣物类型的 Overall 成功率均有提升
- [ ] Unseen 成功率提升 >= 10%
- [ ] 不同衣物类型间无明显干扰

---

## 8. 参考配置与超参

### 8.1 RLinf 官方 Pi0.5 基准配置

来自 `libero_spatial_ppo_openpi_pi05.yaml`:

| 参数 | LIBERO 值 | LeHome 建议值 | 说明 |
|------|-----------|-------------|------|
| total_num_envs (train) | 64 | 8-16 | 布料仿真更慢 |
| total_num_envs (eval) | 500 | 20-40 | |
| max_episode_steps | 240 | 600 | 叠衣任务更长 |
| micro_batch_size | 128 | 8-16 | 受 GPU 显存限制 |
| global_batch_size | 2048 | 64-128 | |
| rollout_epoch | 8 | 2-4 | 减少以适配布料仿真速度 |
| num_action_chunks | 5 | 5-10 | 需与 Pi0.5 chunk_size 匹配 |
| lr | 5e-6 | 5e-6 | 保持不变 |
| value_lr | 1e-4 | 1e-4 | |
| noise_method | flow_sde | flow_sde | |
| train_expert_only | True | True | 冻结 VLM |

### 8.2 GPU 资源规划

| 组件 | GPU 数 | 说明 |
|------|--------|------|
| Env (仿真) | 1-2 | Isaac Lab + 布料粒子 |
| Rollout (推理) | 1 | Pi0.5 模型推理 |
| Actor (训练) | 1-2 | 梯度计算 + 优化器 |
| **总计** | **2-4** | 单机可跑 |

---

## 9. 常见问题与排查

### Q1: pixi 环境中 rlinf 安装失败

```bash
# 尝试直接从源码安装
pixi run pip install -e /home/maozan/code/RLinf
```

### Q2: Ray 无法初始化

```bash
# 单机模式下确保 ray 可用
pixi run ray start --head --num-gpus=4
pixi run python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

### Q3: Isaac Lab 环境在 SubProc 中崩溃

布料仿真的粒子系统在子进程中可能不稳定。尝试:
- 减少 `num_envs`
- 增加 `stabilize_steps`
- 使用 `ignore_terminations: True` 跳过异常终止

### Q4: Pi0.5 forward 与 RLinf 的 openpi 接口不匹配

RLinf 使用自己的 openpi wrapper (`rlinf/models/embodiment/`)，
而我们的 Pi0.5 来自 lerobot。可能需要:
- 用 RLinf 的 checkpoint 转换工具
- 或直接用 lerobot 格式，在 wrapper 中做适配

### Q5: OOM (显存不足)

```yaml
# 降低 batch size
actor:
  micro_batch_size: 4
  global_batch_size: 32

# 启用 offload
rollout:
  enable_offload: True
actor:
  enable_offload: True

# 减少去噪步数
actor:
  model:
    num_steps: 3  # 默认 5, 减少到 3
```

---

## 10. 参考资料

### RLinf 核心

- [RLinf GitHub](https://github.com/RLinf/RLinf)
- [RLinf 文档](https://rlinf.readthedocs.io/en/latest/)
- [RLinf 安装指南](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
- [RLinf 自定义环境教程](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html)

### Pi0.5 + RL

- [RLinf Pi0/Pi0.5 RL 训练文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html)
- [pi_RL: Online RL for Flow-based VLA (arXiv: 2510.25889)](https://arxiv.org/abs/2510.25889)
- [RLinf-VLA (arXiv: 2510.06710)](https://arxiv.org/abs/2510.06710)

### 相关项目

- [ReinFlow: Fine-tuning Flow Policy with RL](https://github.com/ReinFlow/ReinFlow)
- [RLinf-Co: Sim-Real Co-Training](https://arxiv.org/abs/2602.12628)
- [Isaac Lab 环境封装指南](https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html)

### 关键论文

| 论文 | 与本项目的关系 |
|------|-------------|
| pi_RL (2510.25889) | Flow-SDE/Flow-Noise 算法设计，冻结 VLM 只训练 action expert |
| RLinf-VLA (2510.06710) | 统一框架设计，PPO/GRPO 对比实验 |
| SAC Flow | SAC 适配 flow matching policy，sample-efficient |
| RLinf-Co (2602.12628) | Sim-Real co-training，未来可参考 |
