"""Evaluate latent feature separation for trained implicit policy checkpoints.

Compares the trained model's latent space against the raw DINOv2 baseline
on the same set of images, computing cosine similarity, silhouette score,
and Davies-Bouldin index.

Usage:
    # Baseline: raw DINOv2 features
    pixi run python scripts_mz/eval_latent_separation.py

    # Trained checkpoint
    pixi run python scripts_mz/eval_latent_separation.py \
        --checkpoint outputs/train/pi05_implicit_v5/checkpoints/010000

    # Compare before/after side-by-side
    pixi run python scripts_mz/eval_latent_separation.py \
        --checkpoint outputs/train/pi05_implicit_v5/checkpoints/010000 \
        --compare
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Datasets/example/raw_reject_v3_1k6p9_typeprob"
OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis/latent_separation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DISPLAY = ["top_short", "top_long", "pant_short", "pant_long"]
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


def load_episode_class_map():
    with open(DATASET_DIR / "meta/episode_class.json") as f:
        return json.load(f)


def build_video_index(video_dir, num_videos=32):
    import av
    cumsums = []
    for fi in range(num_videos):
        path = f"{video_dir}/file-{fi:03d}.mp4"
        with av.open(path) as c:
            n = c.streams.video[0].frames
        cumsums.append(n + (cumsums[-1] if cumsums else 0))

    def lookup(frame_idx):
        for vi, cum in enumerate(cumsums):
            if frame_idx < cum:
                offset = frame_idx - (cumsums[vi - 1] if vi > 0 else 0)
                return vi, offset
        raise ValueError(f"frame_idx {frame_idx} out of range")
    return lookup


def sample_images(samples_per_class=80):
    import pandas as pd
    import av

    episode_class_map = load_episode_class_map()
    df = pd.read_parquet(DATASET_DIR / "data/chunk-000/file-000.parquet")
    video_dir = DATASET_DIR / "videos/observation.images.top_rgb/chunk-000"
    video_lookup = build_video_index(str(video_dir))
    rng = np.random.RandomState(42)

    class_images = {k: [] for k in CLASS_DISPLAY}

    for class_name in CLASS_DISPLAY:
        eps_of_class = [int(eid) for eid, cn in episode_class_map.items() if cn == class_name]
        eps_set = set(eps_of_class)
        class_df = df[df["episode_index"].isin(eps_set)]
        unique_eps = class_df["episode_index"].unique()
        selected_eps = rng.choice(unique_eps, size=min(samples_per_class, len(unique_eps)), replace=False)

        for ep in selected_eps:
            ep_rows = class_df[class_df["episode_index"] == ep]
            row = ep_rows.iloc[int(len(ep_rows) * 0.5)]
            frame_idx = int(row["index"])
            try:
                vi, offset = video_lookup(frame_idx)
                video_path = f"{video_dir}/file-{vi:03d}.mp4"
                with av.open(video_path) as container:
                    stream = container.streams.video[0]
                    container.seek(int(offset), stream=stream)
                    for frame in container.decode(stream):
                        img = frame.to_ndarray(format="rgb24")
                        class_images[class_name].append(img)
                        break
            except Exception as e:
                print(f"  Skip episode {ep}: {e}")

    return class_images


def compute_metrics(features, labels):
    feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    cos_sim = feats_norm @ feats_norm.T

    intra_vals, inter_vals = [], []
    for ci in range(len(CLASS_DISPLAY)):
        mask = labels == ci
        within = cos_sim[np.ix_(mask, mask)]
        n = within.shape[0]
        if n > 1:
            intra_vals.append((within.sum() - n) / (n * (n - 1)))
        for cj in range(ci + 1, len(CLASS_DISPLAY)):
            mask_j = labels == cj
            cross = cos_sim[np.ix_(mask, mask_j)]
            inter_vals.append(cross.mean())

    overall = cos_sim[np.triu_indices_from(cos_sim, k=1)].mean()
    sil = silhouette_score(features, labels)
    db = davies_bouldin_score(features, labels)

    return {
        "overall_cos_sim": float(overall),
        "intra_cos_sim": float(np.mean(intra_vals)),
        "inter_cos_sim": float(np.mean(inter_vals)),
        "silhouette": float(sil),
        "davies_bouldin": float(db),
    }


def plot_comparison(baseline_feats, baseline_labels, trained_feats, trained_labels, suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for feat, labels, title_prefix, row in [
        (baseline_feats, baseline_labels, "Raw DINOv2", 0),
        (trained_feats, trained_labels, "Trained Latent", 1),
    ]:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(feat)
        var = pca.explained_variance_ratio_

        ax = axes[row, 0]
        for ci, cn in enumerate(CLASS_DISPLAY):
            mask = labels == ci
            ax.scatter(coords[mask, 0], coords[mask, 1], c=COLORS[ci],
                      label=f"{cn} (n={mask.sum()})", alpha=0.6, s=15)
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title(f"{title_prefix} — PCA")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        means = []
        for ci in range(len(CLASS_DISPLAY)):
            mask = labels == ci
            means.append(feat[mask].mean(axis=0))
        means = np.array(means)
        means_n = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
        cos_mat = means_n @ means_n.T

        im = ax.imshow(cos_mat, vmin=0.5, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(CLASS_DISPLAY)))
        ax.set_xticklabels(CLASS_DISPLAY, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(CLASS_DISPLAY)))
        ax.set_yticklabels(CLASS_DISPLAY, fontsize=7)
        ax.set_title(f"{title_prefix} — Class-mean Cosine")
        for i in range(len(CLASS_DISPLAY)):
            for j in range(len(CLASS_DISPLAY)):
                ax.text(j, i, f"{cos_mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Latent Feature Separation: Baseline vs Trained", fontsize=13, fontweight="bold")
    plt.tight_layout()
    outpath = OUTPUT_DIR / f"comparison{suffix}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {outpath}")
    return outpath


def load_trained_encoder(checkpoint_path, device):
    """Load the trained ImplicitConditioner from a checkpoint."""
    from safetensors.torch import load_file
    from scripts.pi05_implicit_policy import PI05ImplicitPolicy, PI05ImplicitConfig
    from scripts.utils.garment_latent_utils import ImplicitConditioningConfig

    # Try loading config + state dict
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = ckpt / "config.json"
    model_path = ckpt / "model.safetensors"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing config.json or model.safetensors in {checkpoint_path}"
        )

    import json
    with open(config_path) as f:
        config_dict = json.load(f)

    # Extract implicit_conditioning from the config
    policy_cfg = config_dict.get("policy", config_dict)
    implicit_raw = policy_cfg.get("implicit_conditioning", {})
    implicit_config = ImplicitConditioningConfig(**implicit_raw)

    # Create encoder
    from scripts.utils.garment_latent_utils import ImplicitConditioner
    conditioner = ImplicitConditioner(implicit_config)

    # Load state dict for implicit_conditioner keys only
    state_dict = load_file(str(model_path))
    implicit_state = {}
    for k, v in state_dict.items():
        if k.startswith("implicit_conditioner."):
            implicit_state[k[len("implicit_conditioner."):]] = v

    conditioner.load_state_dict(implicit_state, strict=False)
    conditioner.to(device)
    conditioner.eval()
    return conditioner


def main():
    parser = argparse.ArgumentParser(description="Evaluate latent feature separation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained checkpoint directory")
    parser.add_argument("--compare", action="store_true",
                       help="Generate side-by-side comparison with baseline")
    parser.add_argument("--samples", type=int, default=80,
                       help="Samples per class (default: 80)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sample images once
    print(f"\nSampling {args.samples} images per class...")
    class_images = sample_images(samples_per_class=args.samples)

    all_images, all_labels = [], []
    for ci, cn in enumerate(CLASS_DISPLAY):
        for img in class_images[cn]:
            all_images.append(img)
            all_labels.append(ci)
    all_labels = np.array(all_labels)
    print(f"Total images: {len(all_images)}")

    # --- Baseline: raw DINOv2 mean-pool ---
    print("\n=== Baseline: Raw DINOv2 (mean pool) ===")
    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                             pretrained=True, num_classes=0, global_pool="").to(device)
    dino.eval()
    baseline_feats = []
    with torch.inference_mode():
        for img in all_images:
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(518, 518), mode="bilinear", align_corners=False).to(device)
            feat = dino.forward_features(t)
            if feat.ndim == 4:
                feat = feat.flatten(2).transpose(1, 2)
            pooled = feat.mean(dim=1)
            baseline_feats.append(pooled.squeeze(0).cpu().numpy())
    baseline_feats = np.array(baseline_feats)

    baseline_metrics = compute_metrics(baseline_feats, all_labels)
    print(f"  overall_cos_sim:    {baseline_metrics['overall_cos_sim']:.4f}")
    print(f"  intra_cos_sim:      {baseline_metrics['intra_cos_sim']:.4f}")
    print(f"  inter_cos_sim:      {baseline_metrics['inter_cos_sim']:.4f}")
    print(f"  silhouette:          {baseline_metrics['silhouette']:.4f}")
    print(f"  davies_bouldin:      {baseline_metrics['davies_bouldin']:.4f}")

    # --- Trained model (if provided) ---
    trained_feats = None
    if args.checkpoint:
        print(f"\n=== Trained Model: {args.checkpoint} ===")
        conditioner = load_trained_encoder(args.checkpoint, device)

        trained_feats = []
        with torch.inference_mode():
            for img in all_images:
                t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                t = F.interpolate(t.unsqueeze(0), size=(518, 518), mode="bilinear", align_corners=False).to(device)
                latent = conditioner.encode({"observation.images.top_rgb": t})
                trained_feats.append(latent.squeeze(0).cpu().numpy())
        trained_feats = np.array(trained_feats)

        trained_metrics = compute_metrics(trained_feats, all_labels)
        print(f"  overall_cos_sim:    {trained_metrics['overall_cos_sim']:.4f}")
        print(f"  intra_cos_sim:      {trained_metrics['intra_cos_sim']:.4f}")
        print(f"  inter_cos_sim:      {trained_metrics['inter_cos_sim']:.4f}")
        print(f"  silhouette:          {trained_metrics['silhouette']:.4f}")
        print(f"  davies_bouldin:      {trained_metrics['davies_bouldin']:.4f}")

        # Delta
        print("\n=== Delta (Trained - Baseline) ===")
        for k in baseline_metrics:
            delta = trained_metrics[k] - baseline_metrics[k]
            direction = "+" if delta >= 0 else ""
            print(f"  {k}: {baseline_metrics[k]:.4f} → {trained_metrics[k]:.4f}  ({direction}{delta:.4f})")

        if args.compare:
            plot_comparison(baseline_feats, all_labels, trained_feats, all_labels)

        # Interpretation
        print("\n=== Interpretation ===")
        sil_delta = trained_metrics['silhouette'] - baseline_metrics['silhouette']
        cos_delta = trained_metrics['inter_cos_sim'] - trained_metrics['intra_cos_sim']
        if trained_metrics['silhouette'] > 0.3:
            print("GOOD: Silhouette > 0.3 — classes are well separated in latent space.")
        elif trained_metrics['silhouette'] > 0.15:
            print("OK: Silhouette 0.15-0.3 — some separation, but still overlap.")
        else:
            print("POOR: Silhouette < 0.15 — classes still heavily overlap.")
        print(f"Intra-Inter gap: {abs(cos_delta):.4f} (> 0.1 = discriminative)")

    else:
        print("\nNo checkpoint provided. Run with --checkpoint to evaluate trained model.")
        # Still generate baseline plot for reference
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(baseline_feats)
        fig, ax = plt.subplots(figsize=(8, 7))
        for ci, cn in enumerate(CLASS_DISPLAY):
            mask = all_labels == ci
            ax.scatter(coords[mask, 0], coords[mask, 1], c=COLORS[ci],
                      label=cn, alpha=0.6, s=20)
        ax.set_title("Raw DINOv2 — PCA (baseline reference)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(OUTPUT_DIR / "baseline_pca.png", dpi=150, bbox_inches="tight")
        print(f"Saved baseline PCA to {OUTPUT_DIR / 'baseline_pca.png'}")


if __name__ == "__main__":
    main()
