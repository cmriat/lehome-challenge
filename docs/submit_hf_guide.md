# LeHome Challenge 提交指南：通过 HuggingFace 上传 pi05 模型

> 适用模型：LeRobot 格式的 pi05 策略
> 提交截止：2026 年 4 月 30 日

---

## 流程概览

```
本地验证模型 → 上传 HuggingFace → 准备提交材料 → 填写 Google Form
```

---

## 第一步：本地验证模型

确保模型能正常加载和推理，避免提交后官方评估失败。

### 1.1 确认模型文件完整性

```bash
ls -lh /home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject/
```

应包含以下文件：

| 文件 | 说明 |
|------|------|
| `config.json` | 模型配置 |
| `model.safetensors` | 模型权重（~7GB） |
| `policy_preprocessor.json` | 预处理配置 |
| `policy_preprocessor_step_2_normalizer_processor.safetensors` | 归一化参数 |
| `policy_postprocessor.json` | 后处理配置 |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | 反归一化参数 |
| `train_config.json` | 训练配置 |

### 1.2 跑一次本地评估

对四种服装类型分别评估，确保模型能正常加载和推理：

```bash
cd /home/nvidia/maoz/lehome-challenge
source .venv/bin/activate

# 长袖上衣
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu \
    --headless

# 短袖上衣
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject \
    --garment_type "top_short" \
    --dataset_root Datasets/example/top_short_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu \
    --headless

# 长裤
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject \
    --garment_type "pant_long" \
    --dataset_root Datasets/example/pant_long_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu \
    --headless

# 短裤
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject \
    --garment_type "pant_short" \
    --dataset_root Datasets/example/pant_short_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu \
    --headless
```

> 确认每次评估都能正常完成，没有模型加载或推理报错。

---

## 第二步：上传到 HuggingFace

### 2.1 安装并登录 HuggingFace CLI

```bash
pip install -U huggingface_hub

# 登录（需要 HF token，在 https://huggingface.co/settings/tokens 获取）
huggingface-cli login
```

### 2.2 创建远程仓库

```bash
# 创建一个 model 类型的 repo（设为 private 可防止他人查看）
huggingface-cli repo create lehome_pi05_step62k_reject --type model
```

### 2.3 上传模型文件

```bash
# 方式一：整目录上传（推荐）
huggingface-cli upload lehome_pi05_step62k_reject \
    /home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject/ \
    --repo-type model

# 方式二：如果上传中断，重新执行同一命令会自动跳过已上传文件
```

上传完成后，模型链接为：
```
https://huggingface.co/<你的HF用户名>/lehome_pi05_step62k_reject
```

### 2.4 验证上传

浏览器打开上面的链接，确认以下文件都已上传：
- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_preprocessor_step_2_normalizer_processor.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`
- `train_config.json`

---

## 第三步：准备提交材料

### 3.1 README.md

创建一个说明文件，告知官方如何评估你的策略：

```markdown
# LeHome Challenge Submission

## Team
- Team Name: <你的团队名>
- Registration ID: <你的注册 ID>

## Policy

- **Type**: LeRobot Policy (pi05)
- **HuggingFace Repo**: https://huggingface.co/<你的HF用户名>/lehome_pi05_step62k_reject

## How to Evaluate

1. 加载 Docker 环境：
   ```bash
   docker run -it --gpus all lehome-challenge
   cd /opt/lehome-challenge
   source .venv/bin/activate
   ```

2. 下载模型：
   ```bash
   pip install -U huggingface_hub
   huggingface-cli download <你的HF用户名>/lehome_pi05_step62k_reject \
       --repo-type model --local-dir checkpoints/pi05
   ```

3. 运行评估（四类服装）：
   ```bash
   for garment in top_long top_short pant_long pant_short; do
       python -m scripts.eval \
           --policy_type lerobot \
           --policy_path checkpoints/pi05 \
           --garment_type "$garment" \
           --dataset_root Datasets/example/${garment}_merged \
           --num_episodes 5 \
           --enable_cameras \
           --device cpu \
           --headless
   done
   ```

## Notes
- 策略为 LeRobot pi05 格式，无需额外依赖
- 模型权重约 7GB（model.safetensors）
- 评估时需要 dataset_root 参数用于加载元数据
```

### 3.2 评估结果 .txt 文件

将本地评估结果保存为文本文件。可以手动整理，格式参考：

```
top_long:
  episode_0: success, steps=XXX
  episode_1: success, steps=XXX
  ...

top_short:
  episode_0: fail, steps=XXX
  ...

pant_long:
  ...

pant_short:
  ...
```

---

## 第四步：填写 Google Form 提交

打开官方提交表单：

**https://docs.google.com/forms/d/e/1FAIpQLSeeFpV4oYxSizNCnplX1Ew--TBafIbVyFg9NPH-hunks2rc7Q/viewform**

按表单要求填写：

| 提交项 | 内容 |
|--------|------|
| README (.md) | 上传第三步准备的 README.md |
| Docker 镜像 URL 或 HF repo 链接 | 填写 `https://huggingface.co/<你的HF用户名>/lehome_pi05_step62k_reject` |
| Rollout 结果 (.txt) | 上传本地评估结果文件 |
| 源代码链接（可选） | 如有 GitHub 仓库可附上 |

---

## 常见问题

**Q: 模型 repo 必须设为 public 吗？**
建议设为 public，方便官方直接下载。如果设为 private，需要确保官方评估账号有访问权限。

**Q: 上传 7GB 的 safetensors 太慢怎么办？**
可以使用断点续传，多次执行 `huggingface-cli upload` 即可，已上传的文件会自动跳过。也可以尝试设置 `HF_HUB_ENABLE_HF_TRANSFER=1` 启用加速传输。

**Q: 官方评估时会用哪种 garment_type？**
评估时服装类型会随机加载，你的策略不应依赖类别标签。如果策略需要类别信息，需自行训练分类器。
