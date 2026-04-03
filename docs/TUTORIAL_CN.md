# LeHome Challenge 2026 完整教程指南

> 从零开始：环境配置 → 数据准备 → 模型训练 → 任务提交

---

## 目录

- [第一部分：环境配置](#第一部分环境配置)
  - [1.1 系统要求](#11-系统要求)
  - [1.2 方式一：Pixi 安装（推荐）](#12-方式一pixi-安装推荐)
  - [1.3 方式二：UV 安装](#13-方式二uv-安装)
  - [1.4 方式三：Docker 安装（推荐服务器/无头环境）](#14-方式三docker-安装推荐服务器无头环境)
- [第二部分：下载官方数据](#第二部分下载官方数据)
- [第三部分：自行仿真收集数据](#第三部分自行仿真收集数据)
  - [3.1 硬件准备（可选）](#31-硬件准备可选)
  - [3.2 键盘遥操作收集数据](#32-键盘遥操作收集数据)
  - [3.3 SO101 Leader 臂收集数据](#33-so101-leader-臂收集数据)
  - [3.4 数据检查与处理](#34-数据检查与处理)
  - [3.5 数据集合并](#35-数据集合并)
- [第四部分：模型训练](#第四部分模型训练)
  - [4.1 训练配置说明](#41-训练配置说明)
  - [4.2 开始训练](#42-开始训练)
  - [4.3 训练监控（WandB）](#43-训练监控wandb)
- [第五部分：模型评估](#第五部分模型评估)
  - [5.1 本地评估（有头服务器）](#51-本地评估有头服务器)
  - [5.2 远程评估（无头服务器）](#52-远程评估无头服务器)
- [第六部分：任务提交](#第六部分任务提交)
- [附录：常见问题](#附录常见问题)

---

## 第一部分：环境配置

### 1.1 系统要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Ubuntu 20.04/22.04 |
| Python | 3.11 |
| GPU | NVIDIA GPU（支持 Isaac Sim 5.1.0） |
| CUDA | 参考 [Isaac Sim 文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) |
| 磁盘空间 | 至少 50GB（含资源和数据集） |

> ⚠️ **重要**：仿真环境当前仅支持 CPU 运行

### 1.2 方式一：Pixi 安装（推荐）

[Pixi](https://pixi.sh/) 是一个现代化的包管理工具，可以一步完成 Python 环境和所有依赖的安装。

#### 步骤 1：安装 Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc  # 或重启终端
```

#### 步骤 2：克隆项目

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

#### 步骤 3：克隆 IsaacLab

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

#### 步骤 4：安装所有依赖

```bash
pixi install
```

> 此命令会自动安装 Python 3.11、Isaac Sim、Isaac Lab、LeHome 及所有依赖，无需其他操作。

#### 步骤 5：安装系统依赖（服务器环境必装）

```bash
sudo apt update
sudo apt install -y \
    libglu1-mesa libgl1 libegl1 \
    libxrandr2 libxinerama1 libxcursor1 \
    libxi6 libxext6 libx11-6

# 设置 NVIDIA 渲染
echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ~/.bashrc
source ~/.bashrc
```

#### 运行命令

Pixi 环境下运行命令有两种方式：

```bash
# 方式 A：使用 pixi run 直接执行
pixi run python -m scripts.eval ...

# 方式 B：进入 pixi shell 后正常执行
pixi shell
python -m scripts.eval ...
```

> 后续章节中的 `source .venv/bin/activate` 可替换为 `pixi shell`，所有 `python` 命令可用 `pixi run python` 替代。

### 1.3 方式二：UV 安装

#### 步骤 1：安装 UV 包管理器

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 或重启终端
```

#### 步骤 2：克隆项目

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

#### 步骤 3：安装 Python 依赖

```bash
uv sync
```

#### 步骤 4：克隆并安装 IsaacLab

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

#### 步骤 5：激活环境并安装 IsaacLab

```bash
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

#### 步骤 6：安装 LeHome 包

```bash
uv pip install -e ./source/lehome
```

#### 步骤 7：安装系统依赖（服务器环境必装）

```bash
sudo apt update
sudo apt install -y \
    libglu1-mesa libgl1 libegl1 \
    libxrandr2 libxinerama1 libxcursor1 \
    libxi6 libxext6 libx11-6

# 设置 NVIDIA 渲染
echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ~/.bashrc
source ~/.bashrc
```

### 1.4 方式三：Docker 安装（推荐服务器/无头环境）

#### 步骤 1：安装 Docker

```bash
# 使用便捷脚本安装
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 配置非 root 用户运行 Docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

# 验证安装
docker run hello-world
```

#### 步骤 2：下载并加载 Docker 镜像

```bash
# 下载镜像（约 15GB，请确保磁盘空间充足）
wget https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz

# 加载镜像
docker load -i lehome-challenge.tar.gz
```

#### 步骤 3：启动容器

```bash
docker run -it --gpus all lehome-challenge
```

#### 步骤 4：容器内激活环境

```bash
cd /opt/lehome-challenge
source .venv/bin/activate
```

---

## 第二部分：下载官方数据

### 2.1 下载仿真资源（必需）

```bash
# 创建 Assets 目录并下载资源
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

资源目录结构：
```
Assets/
├── objects/
│   └── Challenge_Garment/
│       └── Release/        # 比赛用服装
│           ├── Top_Long/
│           ├── Top_Short/
│           ├── Pant_Long/
│           └── Pant_Short/
├── robots/                  # 机器人模型
└── scenes/                  # 场景资源
```

### 2.2 下载示例数据集

#### 完整数据集（推荐）

```bash
# 包含所有四种服装类型的演示数据
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

#### 分离数据集（需要深度信息时）

```bash
# 每种服装单独的数据集，包含深度信息
hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/example
```

数据集目录结构：
```
Datasets/example/
├── top_long_merged/         # 长袖上衣
├── top_short_merged/        # 短袖上衣
├── pant_long_merged/        # 长裤
└── pant_short_merged/       # 短裤
```

---

## 第三部分：自行仿真收集数据

### 3.1 硬件准备（可选）

如需使用 SO101 Leader 臂进行遥操作：

```bash
# 1. 连接设备
# 将双臂通过 USB 连接到电脑

# 2. 查看设备
ls /dev/ttyACM*
# 通常显示: /dev/ttyACM0, /dev/ttyACM1

# 3. 授权
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### 3.2 键盘遥操作收集数据

#### 双臂键盘控制

```bash
source .venv/bin/activate

python -m scripts.dataset_sim record \
    --teleop_device bi-keyboard \
    --task LeHome-BiSO101-Direct-Garment-v2 \
    --garment_name Top_Long_Unseen_0 \
    --garment_version Release \
    --enable_record \
    --num_episode 10 \
    --log_success \
    --device cpu \
    --enable_cameras
```

#### 键盘控制说明

| 按键 | 功能 |
|------|------|
| **B** | 激活遥操作控制（必须先按） |
| **S** | 开始录制当前 episode |
| **N** | 保存当前 episode（标记为成功） |
| **D** | 丢弃当前 episode（重新录制） |
| **ESC** | 中止录制并清空缓冲 |
| **Ctrl+C** | 退出程序 |

**双臂键盘映射：**
- 左臂：T/G, Y/H, U/J, I/K, O/L, Q/A
- 右臂：1/2, 3/4, 5/6, 7/8, 9/0, [/] 或 -/+

### 3.3 SO101 Leader 臂收集数据

#### 首次使用：校准

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --recalibrate \
    --device cpu \
    --enable_cameras
```

校准步骤：
1. 运行命令后会提示校准
2. 将左臂每个关节移动到最大/最小位置
3. 对右臂重复相同操作
4. 按 Ctrl+C 退出（校准数据自动保存）

#### 数据收集

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --garment_name Top_Long_Unseen_0 \
    --garment_version Release \
    --enable_record \
    --num_episode 20 \
    --log_success \
    --step_hz 120 \
    --device cpu \
    --enable_cameras
```

#### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--teleop_device` | 遥操作设备 | `bi-so101leader` / `bi-keyboard` |
| `--garment_name` | 服装名称 | 如 `Top_Long_Unseen_0` |
| `--garment_version` | 资产版本 | `Release`（比赛用） |
| `--num_episode` | 录制 episode 数量 | 20-50 |
| `--step_hz` | 采样频率 | 120 |
| `--log_success` | 记录成功标志 | 建议启用 |
| `--disable_depth` | 禁用深度图 | 加速录制 |
| `--enable_cameras` | 启用相机 | 必须启用 |

### 3.4 数据检查与处理

#### 检查数据集

```bash
# 查看数据集元信息和统计
python -m scripts.dataset inspect \
    --dataset_root Datasets/record/001 \
    --show_stats

# 查看帧数据
python -m scripts.dataset read \
    --dataset_root Datasets/record/001 \
    --num_frames 10
```

#### 回放验证

```bash
python -m scripts.dataset_sim replay \
    --dataset_root Datasets/record/001 \
    --num_replays 1 \
    --disable_depth \
    --device cpu \
    --enable_cameras
```

#### 添加末端执行器位姿（可选）

```bash
python -m scripts.dataset augment \
    --dataset_root Datasets/record/001 \
    --urdf_path Assets/robots/so101_new_calib.urdf \
    --state_unit rad \
    --overwrite
```

#### 移除不需要的特征（节省空间）

```bash
lerobot-edit-dataset \
    --repo_id record_001 \
    --root Datasets/record/001 \
    --new_repo_id record_001_no_depth \
    --operation.type remove_feature \
    --operation.feature_names "['observation.top_depth']"
```

### 3.5 数据集合并

将多个收集会话的数据合并：

```bash
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging

init_logging()

# 要合并的数据集路径
dataset_paths = [
    Path('Datasets/record/001_no_depth'),
    Path('Datasets/record/002_no_depth'),
    Path('Datasets/record/003_no_depth'),
]

# 输出配置
output_dir = Path('Datasets/record/top_long_merged')
datasets = [LeRobotDataset(f'ds_{i:03d}', root=p) for i, p in enumerate(dataset_paths, 1)]

# 执行合并
merged = merge_datasets(datasets, 'top_long_merged', output_dir)
print(f'✓ 合并完成! Episodes: {merged.meta.total_episodes}, Frames: {merged.meta.total_frames}')
"
```

---

## 第四部分：模型训练

### 4.1 训练配置说明

配置文件位于 `configs/` 目录：

| 配置文件 | 策略 | 特点 |
|----------|------|------|
| `train_act.yaml` | ACT | Action Chunking Transformer |
| `train_dp.yaml` | Diffusion Policy | 扩散策略 |
| `train_smolvla.yaml` | SmolVLA | 视觉-语言-动作模型 |

#### 配置文件结构示例

```yaml
# configs/train_act.yaml
dataset:
  repo_id: my_dataset           # 数据集标识
  root: Datasets/record/top_long_merged  # 数据集路径

policy:
  type: act                     # 策略类型
  device: cuda                  # 训练设备
  push_to_hub: false            # 是否推送到 HF Hub

  input_features:
    observation.state:
      type: STATE
      shape: [12]               # 双臂：12维
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.left_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.right_rgb:
      type: VISUAL
      shape: [3, 480, 640]

  output_features:
    action:
      type: ACTION
      shape: [12]

output_dir: outputs/train/act_top_long
batch_size: 16
steps: 30000
save_freq: 5000
log_freq: 1000

wandb:
  enable: false
```

### 4.2 开始训练

#### 使用官方示例数据训练

```bash
source .venv/bin/activate

# 训练 ACT 策略
lerobot-train --config_path=configs/train_act.yaml

# 训练 Diffusion Policy
lerobot-train --config_path=configs/train_dp.yaml

# 训练 SmolVLA
lerobot-train --config_path=configs/train_smolvla.yaml
```

#### 使用自定义数据训练

1. **修改配置文件**

```bash
# 复制配置模板
cp configs/train_act.yaml configs/train_my_act.yaml
```

2. **编辑配置**

```yaml
dataset:
  repo_id: my_custom_dataset
  root: Datasets/record/my_merged_dataset  # 你的数据集路径

output_dir: outputs/train/my_act_model
```

3. **开始训练**

```bash
lerobot-train --config_path=configs/train_my_act.yaml
```

#### 训练参数建议

| 参数 | 小数据集 (<50 episodes) | 中等数据集 (50-200) | 大数据集 (>200) |
|------|------------------------|---------------------|-----------------|
| `batch_size` | 8-16 | 16-32 | 32-64 |
| `steps` | 20000-30000 | 30000-50000 | 50000-100000 |
| `save_freq` | 5000 | 5000-10000 | 10000 |
| `learning_rate` | 1e-4 | 1e-4 | 5e-5 |

### 4.3 训练监控（WandB）

启用 WandB 进行训练可视化：

```yaml
wandb:
  enable: true
  project: lehome-challenge
  entity: your_username  # 可选
```

然后正常启动训练，WandB 会自动记录：
- 损失曲线
- 学习率变化
- 训练进度

---

## 第五部分：模型评估

### 5.1 本地评估（有头服务器）

```bash
source .venv/bin/activate

python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

### 5.2 远程评估（无头服务器）

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos
```

#### 评估参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--policy_type` | 策略类型：`lerobot` 或 `custom` | ✅ |
| `--policy_path` | 模型检查点路径 | ✅ |
| `--dataset_root` | 数据集路径（LeRobot 策略需要） | LeRobot✅ |
| `--garment_type` | 服装类型：`top_long`, `top_short`, `pant_long`, `pant_short`, `custom` | ✅ |
| `--num_episodes` | 每种服装测试 episode 数 | 默认 5 |
| `--max_steps` | 每个 episode 最大步数 | 默认 600 |
| `--headless` | 无头模式（无 GUI） | 服务器必需 |
| `--enable_cameras` | 启用相机渲染 | ✅ |
| `--device` | 仿真设备 | `cpu` |
| `--save_video` | 保存评估视频 | 可选 |
| `--video_dir` | 视频保存目录 | 默认 `outputs/eval_videos` |

### 5.3 评估特定服装

1. **编辑测试列表**

```bash
# 编辑文件，只保留想测试的服装
vim Assets/objects/Challenge_Garment/Release/Release_test_list.txt
```

2. **运行评估**

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type custom \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu \
    --headless
```

---

## 第六部分：任务提交

### 6.1 准备提交材料

1. **模型检查点**

确保模型保存在以下结构：
```
outputs/train/act_top_long/
└── checkpoints/
    └── last/
        └── pretrained_model/
            ├── config.json
            ├── model.safetensors
            └── ...
```

2. **自定义策略（如适用）**

如果使用自定义策略，需要：
- 继承 `BasePolicy` 类
- 实现 `select_action` 方法
- 在 `scripts/eval_policy/__init__.py` 中注册

### 6.2 提交流程

> 📮 详细提交说明请访问 [比赛官网](https://lehome-challenge.com/)

基本流程：
1. 在官网注册账号
2. 创建团队
3. 上传模型检查点
4. 系统自动评估
5. 查看排行榜

### 6.3 评估指标

比赛评估以下指标：
- **成功率**：完成折叠任务的比例
- **效率**：完成任务所需步数
- **泛化能力**：在不同服装上的表现

---

## 附录：常见问题

### Q1: Isaac Sim 启动失败

```bash
# 检查 GPU 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 设置环境变量
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

### Q2: 训练时 GPU 内存不足

```yaml
# 减小 batch_size
batch_size: 8  # 从 16 减小

# 或使用梯度累积
gradient_accumulation_steps: 2
```

### Q3: 无头服务器运行报错

确保添加 `--headless` 参数：
```bash
python -m scripts.eval ... --headless
```

### Q4: 数据集格式不兼容

确保数据集包含以下必需字段：
- `observation.state`
- `action`
- `observation.images.*_rgb`

### Q5: 模型加载失败

检查路径是否指向 `pretrained_model` 目录：
```bash
# 正确
--policy_path outputs/train/act/checkpoints/last/pretrained_model

# 错误
--policy_path outputs/train/act/checkpoints/last/
```

---

## 快速参考命令

```bash
# === 环境激活 ===
# Pixi 方式（推荐）
pixi shell
# UV 方式
source .venv/bin/activate

# === 数据收集 ===
python -m scripts.dataset_sim record \
    --teleop_device bi-keyboard \
    --garment_name Top_Long_Unseen_0 \
    --enable_record --num_episode 10 \
    --device cpu --enable_cameras

# === 数据检查 ===
python -m scripts.dataset inspect --dataset_root Datasets/record/001

# === 训练 ===
lerobot-train --config_path=configs/train_act.yaml

# === 评估（有头） ===
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --enable_cameras --device cpu

# === 评估（无头） ===
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --enable_cameras --device cpu --headless --save_video
```

---

> 📖 更多信息请访问 [LeHome Challenge 官网](https://lehome-challenge.com/)
