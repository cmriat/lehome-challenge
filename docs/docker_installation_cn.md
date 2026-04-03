# Docker 安装指南

本指南提供使用 Docker 安装 LeHome Challenge 环境的分步说明。

## 安装步骤

### 1. 安装 Docker

```bash
# 使用官方便捷脚本安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 配置当前用户免 sudo 使用 Docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

# 验证安装是否成功
docker run hello-world
```

### 2. 下载 Docker 镜像

```bash
wget https://huggingface.co/datasets/lehome/docker/resolve/main/lehome-challenge.tar.gz
```

> **注意：** 下载前请确保有足够的磁盘空间。

### 3. 加载 Docker 镜像

```bash
docker load -i lehome-challenge.tar.gz
```

### 4. 运行容器并激活环境

```bash
# 启动容器（根据需要调整参数）
docker run -it lehome-challenge

# 进入容器后，激活环境并验证
cd /opt/lehome-challenge
source .venv/bin/activate
```

### 5. 评估模型

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu \
    --headless
```

> **注意：** Docker 环境下必须启用无头模式（`--headless`）。

## 更多信息

环境安装完成后，你可以：

- [开始训练](training.md)
- [评估策略](policy_eval.md)
- [返回 README](../README.md)
