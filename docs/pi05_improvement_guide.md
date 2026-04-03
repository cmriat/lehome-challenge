# Pi0.5 模型性能提升指南

本文档整理了在不改变模型架构的前提下，通过数据层面的手段提升 pi0.5 策略性能的三种方法，按投入产出比排序。

**前置说明**：布料仿真只能运行在 CPU 上，所有涉及仿真环境（eval、replay）的命令必须加 `--device cpu`。

---

## 方法一：开启光照与纹理随机化，Replay 扩充数据

### 原理

通过环境域随机化（Domain Randomization），在不改变动作序列的前提下，生成外观更多样的训练数据。Replay 回放已有轨迹时，衣物位姿保持不变（从 `garment_info.json` 精确还原），因此动作序列仍然有效。

### 操作步骤

#### 1. 开启随机化配置

修改 `source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml`：

```yaml
# 桌面纹理随机化（默认关闭，改为 true）
texture_randomization:
  enable: true
  folder: "Assets/textures/surface"
  min_id: 1
  max_id: 100

# 光照随机化（默认关闭，改为 true）
light_randomization:
  enable: true
  prim_path: "/World/Light"
  intensity_range: [3500, 5000]
  color_range: [0.0, 0.2]
```

#### 2. Replay 扩充数据

```bash
python -m scripts.dataset_sim replay \
  --dataset_root Datasets/example/four_types_merged \
  --output_root Datasets/four_types_augmented \
  --num_replays 2 \
  --save_successful_only \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --step_hz 60 \
  --task_description "fold the garment on the table" \
  --garment_cfg_base_path Assets/objects/Challenge_Garment \
  --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
  --disable_depth \
  --device cpu
```

- `--num_replays 2`：每条轨迹回放 2 次，数据量 x2
- `--save_successful_only`：只保留回放成功的（物理模拟有微小随机性，少数可能失败）
- `--disable_depth`：不使用深度观测（与训练配置一致）
- 成功轨迹保存在 `Datasets/four_types_augmented/001/`

#### 3. 合并到训练集

```bash
python -m scripts.utils.dataset_processing merge \
  --source_roots '["Datasets/four_types_augmented", "Datasets/example/four_types_merged"]' \
  --output_root Datasets/example/four_types_merged_v2 \
  --output_repo_id lehome_competition_v2
```

#### 4. 修改训练配置并重新训练

修改 `configs/train_pi05.yaml` 中的 `dataset` 部分：

```yaml
dataset:
  repo_id: lehome_competition_v2
  root: Datasets/example/four_types_merged_v2
```

重新训练：

```bash
accelerate launch --mixed_precision=bf16 --num_processes=8 $(which lerobot-train) \
  --config_path=configs/train_pi05.yaml 2>&1 | tee logs/train_pi05_$(date +%Y%m%d_%H%M%S).log
```

### 预期收益

提升模型对不同桌面纹理和光照条件的鲁棒性，减少过拟合到特定视觉外观的风险。

---

## 方法二：Rejection Sampling（针对性补弱项数据）

### 原理

用当前训练好的 policy 在仿真中进行推理，自动保存成功的 episode，过滤掉失败的。本质上是一种 DAgger 风格的闭环数据增强，可以持续迭代。

### 操作步骤

#### 1. Baseline 评估：确定薄弱 garment 类型

分别评估四种 garment 类型的成功率：

```bash
# Top Long
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step80k \
  --dataset_root Datasets/example/four_types_merged \
  --garment_type top_long \
  --num_episodes 10 \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --task_description "fold the garment on the table" \
  --garment_cfg_base_path Assets/objects/Challenge_Garment \
  --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
  --device cpu

# Top Short
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step80k \
  --dataset_root Datasets/example/four_types_merged \
  --garment_type top_short \
  --num_episodes 10 \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --task_description "fold the garment on the table" \
  --garment_cfg_base_path Assets/objects/Challenge_Garment \
  --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
  --device cpu

# Pant Long
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step80k \
  --dataset_root Datasets/example/four_types_merged \
  --garment_type pant_long \
  --num_episodes 10 \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --task_description "fold the garment on the table" \
  --garment_cfg_base_path Assets/objects/Challenge_Garment \
  --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
  --device cpu

# Pant Short
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step80k \
  --dataset_root Datasets/example/four_types_merged \
  --garment_type pant_short \
  --num_episodes 10 \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --task_description "fold the garment on the table" \
  --garment_cfg_base_path Assets/objects/Challenge_Garment \
  --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
  --device cpu
```

对比四种类型的成功率，确定最薄弱的类型。

#### 2. 针对性 Rejection Sampling

假设 `pant_short` 成功率最低，对其集中采样：

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step80k \
  --dataset_root Datasets/example/four_types_merged \
  --garment_type pant_long \
  --use_random_seed \
  --save_datasets \
  --eval_dataset_path Datasets/eval_pi05_pant_long \
  --num_episodes 100 \
  --max_steps 500 \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --task_description "fold the garment on the table" \
  --garment_cfg_base_path Assets/objects/Challenge_Garment \
  --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
  --device cpu
```

- `--save_datasets`：自动只保存成功的 episode（核心过滤机制）
- `--use_random_seed`：每次 reset 使用随机种子，增加初始条件多样性
- 成功轨迹保存在 `Datasets/eval_pi05_pant_short/001/`，结构与训练数据集一致：

```
Datasets/eval_pi05_pant_short/001/
  data/chunk-000/file-000.parquet          # 帧数据（state, action 等）
  videos/observation.images.top_rgb/       # 顶部相机视频
  videos/observation.images.left_rgb/      # 左腕相机视频
  videos/observation.images.right_rgb/     # 右腕相机视频
  meta/
    info.json                               # 数据集元信息（总 episodes、总 frames、fps 等）
    garment_info.json                       # 每个 episode 的衣物初始位姿（xyz + 欧拉角）
    episodes/chunk-000/file-000.parquet     # episode 索引
    tasks.parquet                           # 任务描述
```

**采集数据量估算**（以 pant_short 为例，12 种 garment，每种 100 episodes）：

| 成功率 | 成功 episodes | 预估帧数（~266帧/episode） |
|--------|--------------|--------------------------|
| 5% | 60 条 | ~16,000 帧 |
| 10% | 120 条 | ~32,000 帧 |
| 20% | 240 条 | ~64,000 帧 |

原始数据集为 1000 条 / 265,798 帧，成功率 10% 时新增约 12% 数据量。如成功率过低，可增大 `--num_episodes`（如 200 或 500）。

#### 3. 合并到训练集

```bash
python -m scripts.utils.dataset_processing merge \
  --source_roots '["Datasets/eval_pi05_pant_short/001", "Datasets/example/four_types_merged"]' \
  --output_root Datasets/example/four_types_merged_v2 \
  --output_repo_id lehome_competition_v2
```

#### 4. 重新训练并迭代

修改 `configs/train_pi05.yaml` 中的 `dataset` 部分：

```yaml
dataset:
  repo_id: lehome_competition_v2
  root: Datasets/example/four_types_merged_v2
```

重新训练：

```bash
accelerate launch --mixed_precision=bf16 --num_processes=8 $(which lerobot-train) \
  --config_path=configs/train_pi05.yaml 2>&1 | tee logs/train_pi05_$(date +%Y%m%d_%H%M%S).log
```

训练完成后可用新模型重复以上步骤，形成多轮迭代闭环。

### 进阶：扩大衣物初始位姿范围

Rejection Sampling 时可以搭配扩大衣物初始位姿的随机化范围，让 policy 在更多样的初始条件下探索。修改 `particle_garment_cfg.yaml`：

```yaml
initial_pos_range: [-0.06, -0.08, 0.72, 0.06, 0.08, 0.74]
initial_rot_range: [-30, -50, -10, 30, 50, 10]
soft_reset_pos_range: [-0.06, -0.08, 0.72, 0.06, 0.08, 0.74]
soft_reset_rot_range: [-30, -50, -10, 30, 50, 10]
```

> **注意**：扩大位姿范围**不能通过 replay 来扩充数据**（replay 使用录制时的原始位姿，动作序列对新位姿无效）。只能通过 policy 推理（rejection sampling）或重新遥操录制来利用。

### 预期收益

直接补齐薄弱 garment 类型的数据短板，提升整体成功率。多轮迭代可持续提升。

---

## 方法三：启用 LeRobot 框架的在线图像增强

### 原理

LeRobot 框架部分策略支持训练时的在线图像变换（如 Color Jitter、Random Grayscale 等），可提升视觉鲁棒性。需要先确认当前安装的 lerobot 版本中 pi0.5 是否支持。

### 操作步骤

#### 1. 检查 pi0.5 是否支持在线增强

```bash
python -c "from lerobot.configs.policies import Pi05Config; import inspect; print(inspect.getsource(Pi05Config))"
```

查看输出中是否有 `image_transform`、`augmentation` 或类似的配置字段。

#### 2. 如果支持，在配置中启用

在 `configs/train_pi05.yaml` 的 `policy` 下添加对应的增强配置（具体字段名取决于上一步的检查结果）。

#### 3. 如果不支持

可考虑以下替代方案：
- 在数据预处理阶段（离线）对图像应用增强，生成增强后的数据集副本
- 切换到支持在线增强的策略（如 Diffusion Policy 的 `crop_is_random`）

### 预期收益

提升模型对颜色、亮度变化的鲁棒性，减少对特定视觉条件的过拟合。

---

## 组合建议

三种方法可以组合使用，推荐顺序：

1. **先做方法一**（光照+纹理 replay）— 成本最低，改动最小
2. **Baseline 评估** → 确定薄弱项
3. **再做方法二**（针对性 rejection sampling）— 收益最高
4. **最后检查方法三**（在线增强）— 取决于框架支持

> **重要提醒**：多轮迭代时务必保留原始遥操数据集作为 base，只 append 新数据，避免策略在自身生成的数据分布上过拟合。
