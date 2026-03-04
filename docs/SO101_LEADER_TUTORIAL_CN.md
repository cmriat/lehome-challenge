# SO101 Leader 臂数据采集完整教程

> 从硬件准备到数据采集的全流程指南

---

## 目录

- [第一部分：硬件准备](#第一部分硬件准备)
- [第二部分：系统配置](#第二部分系统配置)
- [第三部分：首次校准](#第三部分首次校准)
- [第四部分：验证连接](#第四部分验证连接)
- [第五部分：数据采集](#第五部分数据采集)
- [第六部分：数据检查](#第六部分数据检查)
- [第七部分：常见问题](#第七部分常见问题)

---

## 第一部分：硬件准备

### 1.1 硬件清单

| 组件 | 数量 | 说明 |
|------|------|------|
| SO101 Leader 臂 | 2 个 | 左右各一个 |
| USB 数据线 | 2 根 | 连接电脑 |
| 电源适配器 | 2 个 | 为 Leader 臂供电 |
| 电脑 | 1 台 | Ubuntu 20.04/22.04 |

### 1.2 硬件购买建议

**SO101 Leader 臂** 可以从以下渠道购买：

| 渠道 | 链接 | 价格参考 |
|------|------|----------|
| 官方渠道 | [TheRobotStudio](https://www.therobotstudio.com/) | $100-150/个 |
| 淘宝/天猫 | 搜索 "SO101 机械臂" | ¥500-800/个 |
| 亚马逊 | 搜索 "SO101 robot arm" | $100-200/个 |

> 💡 **提示**：建议购买**双臂套装**，本项目使用双臂操作

### 1.3 硬件连接

```
┌─────────────────────────────────────────────────────────┐
│                      连接示意图                          │
│                                                         │
│   ┌──────────┐         ┌──────────┐                     │
│   │ 左臂      │         │ 右臂      │                     │
│   │ (Leader) │         │ (Leader) │                     │
│   └────┬─────┘         └────┬─────┘                     │
│        │                    │                           │
│        │ USB                │ USB                       │
│        │                    │                           │
│        ▼                    ▼                           │
│   ┌─────────────────────────────────────┐               │
│   │           电脑 (Ubuntu)              │               │
│   │    /dev/ttyACM0      /dev/ttyACM1   │               │
│   └─────────────────────────────────────┘               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**连接步骤：**

1. **先连接左臂**
   - 将左臂 USB 插入电脑
   - 接通左臂电源
   - 等待设备识别（约 2-3 秒）

2. **再连接右臂**
   - 将右臂 USB 插入电脑
   - 接通右臂电源
   - 等待设备识别

3. **验证连接**
   ```bash
   ls /dev/ttyACM*
   # 应该显示: /dev/ttyACM0 /dev/ttyACM1
   ```

---

## 第二部分：系统配置

### 2.1 检查设备连接

```bash
# 查看串口设备
ls -la /dev/ttyACM*

# 预期输出：
# crw-rw---- 1 root dialout ... /dev/ttyACM0
# crw-rw---- 1 root dialout ... /dev/ttyACM1
```

### 2.2 配置串口权限

**方法一：临时授权（每次重启后需重新执行）**

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

**方法二：永久授权（推荐）**

```bash
# 将当前用户添加到 dialout 组
sudo usermod -aG dialout $USER

# 注销并重新登录使生效
# 或者执行：
newgrp dialout

# 验证权限
groups $USER
# 输出应包含: dialout
```

### 2.3 确定左右臂映射

```bash
# 方法一：通过插拔顺序
# 先插入的设备 → /dev/ttyACM0（通常作为左臂）
# 后插入的设备 → /dev/ttyACM1（通常作为右臂）

# 方法二：通过测试命令（后续校准时验证）
# 如果发现左右臂反了，交换端口号即可
```

### 2.4 激活项目环境

```bash
cd ~/maoz/lehome-challenge
source .venv/bin/activate
```

---

## 第三部分：首次校准

> ⚠️ **重要**：首次使用或更换硬件后必须校准！

### 3.1 启动校准程序

```bash
cd ~/maoz/lehome-challenge
source .venv/bin/activate

python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --recalibrate \
    --device cpu \
    --enable_cameras
```

### 3.2 校准流程详解

启动后，终端会显示类似信息：

```
[INFO] Starting calibration for left arm...
[INFO] Please move the arm to record joint limits.
```

**校准步骤（对每个臂重复）：**

#### Step 1: 肩部旋转 (Shoulder Pan)

```
┌─────────────────────────────────────┐
│  1. 肩部左右旋转                      │
│                                     │
│     ┌───┐                           │
│     │ ● │ ← 肩部关节                 │
│     └─┬─┘                           │
│       │                             │
│       │ 缓慢左右转动到极限位置        │
│       │                             │
│     ──┴──                           │
│    ←   →  来回移动 2-3 次            │
└─────────────────────────────────────┘
```

#### Step 2: 肩部俯仰 (Shoulder Lift)

```
┌─────────────────────────────────────┐
│  2. 肩部上下摆动                      │
│                                     │
│       ┌───┐                         │
│       │ ● │ ← 肩部关节               │
│       └─┬─┘                         │
│         │                           │
│        ┌┴┐                          │
│        └─┘ ← 缓慢上下移动            │
│                                     │
│        ↑   ↓  来回移动 2-3 次        │
└─────────────────────────────────────┘
```

#### Step 3: 肘部弯曲 (Elbow Flex)

```
┌─────────────────────────────────────┐
│  3. 肘部弯曲                         │
│                                     │
│         ┌───┐                       │
│         │   │                       │
│         └─┬─┘                       │
│           │                         │
│         ┌─●─┐ ← 肘部关节             │
│         └─┬─┘                       │
│           │                         │
│      伸直 ↔ 弯曲  来回 2-3 次        │
└─────────────────────────────────────┘
```

#### Step 4-5: 腕部关节 (Wrist Flex & Roll)

```
┌─────────────────────────────────────┐
│  4. 腕部上下摆动 (Wrist Flex)        │
│  5. 腕部旋转 (Wrist Roll)           │
│                                     │
│         ┌───┐                       │
│         │   │                       │
│         └───┘                       │
│           │                         │
│         ┌─●─┐ ← 腕部关节             │
│         └─┬─┘                       │
│           │                         │
│     上下摆动 + 左右旋转              │
└─────────────────────────────────────┘
```

#### Step 6: 夹爪 (Gripper)

```
┌─────────────────────────────────────┐
│  6. 夹爪开合                         │
│                                     │
│         ┌───┐                       │
│         │   │                       │
│         └───┘                       │
│           │                         │
│         ╔═══╗                       │
│         ║   ║ ← 夹爪                 │
│         ╚═╦═╝                       │
│           │                         │
│     打开 ↔ 闭合  来回 2-3 次         │
└─────────────────────────────────────┘
```

### 3.3 校准完成

校准数据会自动保存到：

```
source/lehome/lehome/devices/lerobot/.cache/
├── left_arm_calibration.json   # 左臂校准数据
└── right_arm_calibration.json  # 右臂校准数据
```

查看校准文件：

```bash
cat source/lehome/lehome/devices/lerobot/.cache/left_arm_calibration.json
```

示例内容：

```json
{
  "shoulder_pan": {"min": -2.5, "max": 2.5},
  "shoulder_lift": {"min": -1.5, "max": 1.5},
  "elbow_flex": {"min": -2.8, "max": 2.8},
  "wrist_flex": {"min": -2.0, "max": 2.0},
  "wrist_roll": {"min": -3.0, "max": 3.0},
  "gripper": {"min": 0.0, "max": 0.04}
}
```

---

## 第四部分：验证连接

### 4.1 启动验证程序

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --device cpu \
    --enable_cameras
```

### 4.2 验证步骤

1. **等待仿真窗口打开**
   - Isaac Sim 窗口会显示虚拟场景
   - 虚拟机器人（SO101 双臂）出现在场景中

2. **测试左臂**
   - 缓慢移动物理左臂
   - 观察虚拟机器人的左臂是否同步跟随

3. **测试右臂**
   - 缓慢移动物理右臂
   - 观察虚拟机器人的右臂是否同步跟随

4. **检查映射是否正确**
   - 物理左臂 → 虚拟左臂 ✅
   - 物理右臂 → 虚拟右臂 ✅

### 4.3 如果映射错误

```bash
# 如果左右臂反了，交换端口号
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM1 \   # 交换
    --right_arm_port /dev/ttyACM0 \  # 交换
    --device cpu \
    --enable_cameras
```

### 4.4 退出验证

按 `Ctrl+C` 退出程序

---

## 第五部分：数据采集

### 5.1 采集前准备

**检查清单：**

- [ ] 环境已激活 (`source .venv/bin/activate`)
- [ ] Leader 臂已连接并通电
- [ ] 串口权限已配置
- [ ] 校准已完成
- [ ] 仿真资源已下载 (`Assets/` 目录存在)

### 5.2 启动数据采集

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

### 5.3 参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `--teleop_device` | `bi-so101leader` | 双臂 SO101 Leader 设备 |
| `--left_arm_port` | `/dev/ttyACM0` | 左臂串口 |
| `--right_arm_port` | `/dev/ttyACM1` | 右臂串口 |
| `--garment_name` | `Top_Long_Unseen_0` | 服装名称 |
| `--garment_version` | `Release` | 资产版本（比赛用 Release） |
| `--enable_record` | - | 启用录制（必须） |
| `--num_episode` | `20` | 计划录制的 episode 数量 |
| `--log_success` | - | 记录成功标志 |
| `--step_hz` | `120` | 采样频率（推荐 120Hz） |
| `--device` | `cpu` | 仿真设备（必须用 CPU） |
| `--enable_cameras` | - | 启用相机（必须） |

### 5.4 采集操作流程

```
┌─────────────────────────────────────────────────────────┐
│                    采集操作流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 启动程序，等待仿真窗口打开                            │
│     │                                                   │
│     ▼                                                   │
│  2. 按 [B] 键激活遥操作控制                              │
│     │  → 现在可以控制虚拟机器人了                        │
│     ▼                                                   │
│  3. 移动 Leader 臂，让机器人到达初始位置                  │
│     │                                                   │
│     ▼                                                   │
│  4. 按 [S] 键开始录制当前 episode                        │
│     │  → 屏幕显示 "Recording..."                        │
│     ▼                                                   │
│  5. 执行折叠任务演示                                     │
│     │  → 移动 Leader 臂完成折叠操作                      │
│     │                                                   │
│     ▼                                                   │
│  6. 完成后按 [N] 键保存 episode（标记为成功）             │
│     │  或按 [D] 键丢弃当前 episode（重新录制）            │
│     │                                                   │
│     ▼                                                   │
│  7. 重复步骤 3-6，直到录制足够数量的 episodes            │
│     │                                                   │
│     ▼                                                   │
│  8. 按 [Ctrl+C] 退出程序                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 5.5 快捷键速查

| 按键 | 功能 | 说明 |
|------|------|------|
| **B** | 激活控制 | 必须先按此键才能控制机器人 |
| **S** | 开始录制 | 开始记录当前 episode |
| **N** | 保存成功 | 保存当前 episode 并标记为成功 |
| **D** | 丢弃重录 | 丢弃当前 episode，重新录制 |
| **ESC** | 中止清空 | 中止当前录制，清空缓冲 |
| **Ctrl+C** | 退出程序 | 退出整个程序 |

### 5.6 采集技巧

#### 技巧 1：稳定的初始位置

```
每次 episode 开始前：
1. 将机器人移动到相同的初始位置
2. 等待 1-2 秒让服装稳定
3. 按 S 开始录制
```

#### 技巧 2：平滑的动作

```
❌ 避免：快速、突然的动作
   → 会导致数据抖动，影响训练

✅ 推荐：缓慢、平滑的动作
   → 像在演示给新手看一样
```

#### 技巧 3：失败时及时丢弃

```
如果操作失误：
1. 立即按 D 丢弃当前 episode
2. 不要勉强继续
3. 重新开始新的 episode
```

#### 技巧 4：保持一致性

```
✅ 每次使用相同的：
   - 初始位置
   - 操作顺序
   - 折叠方式

这样可以让策略学习更稳定
```

### 5.7 采集不同服装

```bash
# 长袖上衣
--garment_name Top_Long_Unseen_0

# 短袖上衣
--garment_name Top_Short_Unseen_0

# 长裤
--garment_name Pant_Long_Unseen_0

# 短裤
--garment_name Pant_Short_Unseen_0
```

查看可用服装：

```bash
ls Assets/objects/Challenge_Garment/Release/
# Top_Long/  Top_Short/  Pant_Long/  Pant_Short/

# 查看具体服装列表
cat Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt
```

---

## 第六部分：数据检查

### 6.1 查看数据集位置

数据默认保存在：

```
Datasets/record/
└── 001/                    # 第一次采集
    ├── data/               # 状态和动作数据
    ├── images/             # 相机图像
    ├── videos/             # 视频文件
    ├── meta/               # 元数据
    │   ├── info.json
    │   ├── garment_info.json
    │   └── stats.json
    └── pointclouds/        # 点云数据（可选）
```

### 6.2 检查数据集信息

```bash
python -m scripts.dataset inspect \
    --dataset_root Datasets/record/001 \
    --show_stats
```

输出示例：

```
Dataset: Datasets/record/001
=====================================
Total Episodes: 20
Total Frames: 8456
Total Seconds: 70.5

Features:
  - observation.state: (12,) float32
  - action: (12,) float32
  - observation.images.top_rgb: (480, 640, 3) uint8
  - observation.images.left_rgb: (480, 640, 3) uint8
  - observation.images.right_rgb: (480, 640, 3) uint8

Statistics:
  observation.state:
    mean: [0.12, -0.34, 0.56, ...]
    std: [0.23, 0.45, 0.12, ...]
```

### 6.3 查看帧数据

```bash
python -m scripts.dataset read \
    --dataset_root Datasets/record/001 \
    --num_frames 5
```

### 6.4 回放验证

```bash
python -m scripts.dataset_sim replay \
    --dataset_root Datasets/record/001 \
    --num_replays 1 \
    --device cpu \
    --enable_cameras
```

---

## 第七部分：常见问题

### Q1: 找不到设备 /dev/ttyACM*

```bash
# 检查 USB 连接
lsusb

# 检查内核消息
dmesg | tail -20

# 可能的原因：
# 1. USB 线未插好
# 2. 驱动问题
# 3. 设备未通电
```

### Q2: 权限被拒绝

```bash
# 错误信息: Permission denied: '/dev/ttyACM0'

# 解决方案 1: 临时授权
sudo chmod 666 /dev/ttyACM0

# 解决方案 2: 永久授权
sudo usermod -aG dialout $USER
# 然后注销重新登录
```

### Q3: 机器人不动或抖动

```bash
# 可能原因 1: 校准问题
# 解决: 重新校准
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --recalibrate \
    --device cpu \
    --enable_cameras

# 可能原因 2: 左右臂映射错误
# 解决: 交换端口号
--left_arm_port /dev/ttyACM1 \
--right_arm_port /dev/ttyACM0
```

### Q4: 仿真窗口很卡

```bash
# 确保使用 CPU 设备
--device cpu

# 关闭不必要的程序释放内存

# 检查 GPU 驱动
nvidia-smi
```

### Q5: 录制时程序崩溃

```bash
# 检查磁盘空间
df -h

# 检查内存
free -h

# 查看日志
# 日志通常在终端输出
```

### Q6: 数据集格式问题

```bash
# 验证数据集完整性
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('test', root='Datasets/record/001')
print(f'Episodes: {ds.meta.total_episodes}')
print(f'Frames: {ds.meta.total_frames}')
"
```

### Q7: 如何删除校准数据重新开始

```bash
# 删除校准文件
rm source/lehome/lehome/devices/lerobot/.cache/left_arm_calibration.json
rm source/lehome/lehome/devices/lerobot/.cache/right_arm_calibration.json

# 重新运行校准
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --recalibrate \
    --device cpu \
    --enable_cameras
```

---

## 快速参考命令

```bash
# === 权限配置 ===
sudo usermod -aG dialout $USER
newgrp dialout

# === 激活环境 ===
cd ~/maoz/lehome-challenge
source .venv/bin/activate

# === 校准（首次使用） ===
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --recalibrate \
    --device cpu \
    --enable_cameras

# === 验证连接 ===
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --device cpu \
    --enable_cameras

# === 数据采集 ===
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

# === 数据检查 ===
python -m scripts.dataset inspect --dataset_root Datasets/record/001 --show_stats

# === 回放验证 ===
python -m scripts.dataset_sim replay \
    --dataset_root Datasets/record/001 \
    --device cpu \
    --enable_cameras
```

---

> 📖 返回 [主教程](TUTORIAL_CN.md) | [数据集处理指南](datasets.md)
