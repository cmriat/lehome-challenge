# LeHome Challenge 2026 Submission

## Team

- **Team Name**: `Hajimi`
- **Registration ID**: `r162`

## Policy

- **Type**: LeRobot Policy (pi05)
- **HuggingFace Repo**: https://huggingface.co/cat314/lehome_pi05
- **Checkpoint**: pi05_step75k

## Local Evaluation Results

| Garment Type | Episodes | Success Rate | Avg Return |
|---|---|---|---|
| Top Long | 240 | 82.08% | 142.03 +/- 17.10 |
| Top Short | 240 | 71.67% | 170.49 +/- 36.46 |
| Pant Long | 240 | 69.58% | 126.27 +/- 12.43 |
| Pant Short | 240 | 87.08% | 144.84 +/- 21.46 |
| **Overall** | **960** | **77.60%** | **145.91** |

## How to Evaluate

### 1. Start Docker Environment

```bash
docker run -it --gpus all --shm-size=16gb --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash lehome-challenge
```

```bash
cd /opt/lehome-challenge
source .venv/bin/activate
```

### 2. Install Required Dependency

The official Docker image requires a patched version of transformers for pi05 support:

```bash
python -m ensurepip && python -m pip install git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi
```

### 3. Download Model from HuggingFace

```bash
pip install -U huggingface_hub
huggingface-cli download cat314/lehome_pi05 \
    --repo-type model --local-dir checkpoints/pi05
```

### 4. Run Evaluation (All 4 Garment Types)

```bash
for garment in top_long top_short pant_long pant_short; do
    python -m scripts.eval \
        --policy_type lerobot \
        --policy_path checkpoints/pi05 \
        --garment_type "$garment" \
        --dataset_root Datasets/example/${garment}_merged \
        --num_episodes 10 \
        --max_steps 600 \
        --task LeHome-BiSO101-Direct-Garment-v2 \
        --task_description "fold the garment on the table" \
        --enable_cameras \
        --device cpu \
        --headless
done
```

## Notes

- Policy is in LeRobot pi05 format
- Model weights are ~7GB (`model.safetensors`)
- The `dataset_root` parameter is required for loading metadata (normalization stats)
- Must install the patched transformers (`fix/lerobot_openpi` branch) before evaluation
- Evaluation uses CPU inference (`--device cpu`)
