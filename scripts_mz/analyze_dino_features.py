"""DINOv2 feature analysis on top-camera garment images.

Samples frames from 4 garment types, runs DINOv2, and visualizes the embedding
space via PCA / cosine-similarity heatmap.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Datasets/example/raw_reject_v3_1k6p9_typeprob"
OUTPUT_DIR = PROJECT_ROOT / "outputs/analysis/dino_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_DISPLAY = ["top_short", "top_long", "pant_short", "pant_long"]


def load_episode_class_map():
    with open(DATASET_DIR / "meta/episode_class.json") as f:
        return json.load(f)


def build_video_index(video_dir, num_videos=32):
    """Build a mapping: global_frame_index -> (video_file_idx, offset_in_video)."""
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    episode_class_map = load_episode_class_map()
    print(f"Loaded {len(episode_class_map)} episode class mappings")

    import pandas as pd

    parquet_path = DATASET_DIR / "data/chunk-000/file-000.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from parquet")

    # Build video file index mapping
    video_dir = DATASET_DIR / "videos/observation.images.top_rgb/chunk-000"
    video_lookup = build_video_index(str(video_dir))
    print("Built video index mapping")

    # Sample frames per class
    samples_per_class = 80
    rng = np.random.RandomState(42)
    class_frame_indices = {k: [] for k in CLASS_DISPLAY}

    for class_name in CLASS_DISPLAY:
        eps_of_class = [int(eid) for eid, cn in episode_class_map.items() if cn == class_name]
        eps_set = set(eps_of_class)
        class_df = df[df["episode_index"].isin(eps_set)]
        unique_eps = class_df["episode_index"].unique()
        selected_eps = rng.choice(
            unique_eps, size=min(samples_per_class, len(unique_eps)), replace=False
        )

        for ep in selected_eps:
            ep_rows = class_df[class_df["episode_index"] == ep]
            n = len(ep_rows)
            mid_idx = int(n * 0.5)
            row = ep_rows.iloc[mid_idx]
            frame_idx = int(row["index"])
            try:
                vi, offset = video_lookup(frame_idx)
                class_frame_indices[class_name].append((vi, offset))
            except ValueError as e:
                print(f"  Skip episode {ep} frame {frame_idx}: {e}")

    # Extract images from videos
    print("\nExtracting frames from videos...")
    import av

    class_images = {k: [] for k in CLASS_DISPLAY}

    for class_name in CLASS_DISPLAY:
        for vi, offset in class_frame_indices[class_name]:
            video_path = f"{video_dir}/file-{vi:03d}.mp4"
            try:
                with av.open(video_path) as container:
                    stream = container.streams.video[0]
                    container.seek(int(offset), stream=stream)
                    for frame in container.decode(stream):
                        img = frame.to_ndarray(format="rgb24")
                        class_images[class_name].append(img)
                        break
            except Exception as e:
                print(f"  Error {video_path} offset {offset}: {e}")

    for class_name in CLASS_DISPLAY:
        print(f"  {class_name}: {len(class_images[class_name])} images")
    total_images = sum(len(v) for v in class_images.values())
    print(f"  Total: {total_images} images")

    if total_images == 0:
        print("ERROR: No images extracted!")
        return

    # Load DINOv2
    print("\nLoading DINOv2...")
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        global_pool="",
    ).to(device)
    model.eval()
    print(f"DINOv2 loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Run inference
    print("Running DINOv2 inference...")
    all_features = []
    all_labels = []

    with torch.inference_mode():
        for class_idx, class_name in enumerate(CLASS_DISPLAY):
            for img in class_images[class_name]:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = F.interpolate(
                    img_t.unsqueeze(0), size=(518, 518), mode="bilinear", align_corners=False
                )
                img_t = img_t.to(device)

                feat = model.forward_features(img_t)
                if feat.ndim == 4:
                    feat = feat.flatten(2).transpose(1, 2)
                pooled = feat.mean(dim=1)
                all_features.append(pooled.squeeze(0).cpu().numpy())
                all_labels.append(class_idx)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    print(f"Features shape: {all_features.shape}")

    # === PCA 2D ===
    print("\nComputing PCA...")
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(all_features)
    var_explained = pca.explained_variance_ratio_

    # === Cosine similarity ===
    print("\nCosine similarity analysis:")
    feats_norm = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)
    cos_sim = feats_norm @ feats_norm.T

    for class_idx, class_name in enumerate(CLASS_DISPLAY):
        mask = all_labels == class_idx
        within = cos_sim[np.ix_(mask, mask)]
        n_within = within.shape[0]
        if n_within > 1:
            within_mean = (within.sum() - n_within) / (n_within * (n_within - 1))
        else:
            within_mean = 1.0
        print(f"  {class_name}: intra-class cosine sim = {within_mean:.4f}")

    inter_vals = []
    for i in range(len(CLASS_DISPLAY)):
        for j in range(i + 1, len(CLASS_DISPLAY)):
            mask_i = all_labels == i
            mask_j = all_labels == j
            cross = cos_sim[np.ix_(mask_i, mask_j)]
            mean_cross = cross.mean()
            inter_vals.append(mean_cross)
            print(f"  {CLASS_DISPLAY[i]} vs {CLASS_DISPLAY[j]}: cross = {mean_cross:.4f}")

    print(f"  Overall inter-class mean: {np.mean(inter_vals):.4f}")
    overall_mean = cos_sim[np.triu_indices_from(cos_sim, k=1)].mean()
    print(f"  Overall pairwise cosine sim: {overall_mean:.4f}")

    # === Plotting ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for class_idx, class_name in enumerate(CLASS_DISPLAY):
        mask = all_labels == class_idx
        axes[0].scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[class_idx],
            label=f"{class_name} (n={mask.sum()})",
            alpha=0.6,
            s=20,
        )

    axes[0].set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    axes[0].set_title("DINOv2 Feature Space — PCA (Mean Pooling)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Cosine heatmap
    class_mean_feats = []
    for class_idx in range(len(CLASS_DISPLAY)):
        mask = all_labels == class_idx
        class_mean_feats.append(all_features[mask].mean(axis=0))
    class_mean_feats = np.array(class_mean_feats)
    class_mean_norm = class_mean_feats / (np.linalg.norm(class_mean_feats, axis=1, keepdims=True) + 1e-8)
    class_cos_sim = class_mean_norm @ class_mean_norm.T

    im = axes[1].imshow(class_cos_sim, vmin=0.7, vmax=1.0, cmap="RdYlGn")
    axes[1].set_xticks(range(len(CLASS_DISPLAY)))
    axes[1].set_xticklabels(CLASS_DISPLAY, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(CLASS_DISPLAY)))
    axes[1].set_yticklabels(CLASS_DISPLAY, fontsize=8)
    axes[1].set_title("Class-mean Cosine Similarity (DINOv2 mean-pool features)")
    for i in range(len(CLASS_DISPLAY)):
        for j in range(len(CLASS_DISPLAY)):
            axes[1].text(j, i, f"{class_cos_sim[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.suptitle(
        f"DINOv2 Feature Analysis — Top Camera Garment Images\n"
        f"mean pairwise cos-sim={overall_mean:.4f}, PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}",
        fontsize=12,
    )
    plt.tight_layout()
    outpath = OUTPUT_DIR / "dino_feature_analysis.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {outpath}")

    # === 3D plot ===
    pca3 = PCA(n_components=3, random_state=42)
    features_3d = pca3.fit_transform(all_features)

    fig3d = plt.figure(figsize=(12, 10))
    ax3d = fig3d.add_subplot(111, projection="3d")
    for class_idx, class_name in enumerate(CLASS_DISPLAY):
        mask = all_labels == class_idx
        ax3d.scatter(
            features_3d[mask, 0],
            features_3d[mask, 1],
            features_3d[mask, 2],
            c=colors[class_idx],
            label=class_name,
            alpha=0.7,
            s=25,
        )
    ax3d.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]:.1%})")
    ax3d.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]:.1%})")
    ax3d.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]:.1%})")
    ax3d.set_title("DINOv2 Features — PCA 3D")
    ax3d.legend()
    outpath3d = OUTPUT_DIR / "dino_feature_3d.png"
    plt.savefig(outpath3d, dpi=150, bbox_inches="tight")
    print(f"Saved to {outpath3d}")

    # === Metrics ===
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    sil = silhouette_score(all_features, all_labels)
    db = davies_bouldin_score(all_features, all_labels)
    print(f"\nSilhouette Score: {sil:.4f} (range [-1, 1], higher = better)")
    print(f"Davies-Bouldin Index: {db:.4f} (lower = better)")

    # === Summary ===
    print("\n=== Summary ===")
    if overall_mean > 0.95:
        print("RESULT: EXTREMELY COLLAPSED (> 0.95) — DINOv2 sees all garments nearly identically.")
    elif overall_mean > 0.9:
        print("RESULT: HIGHLY COLLAPSED — all pairwise cosine sim > 0.9")
        print("        DINOv2 features are nearly identical across garment types.")
    elif sil < 0.1:
        print("RESULT: POORLY SEPARATED — silhouette near 0, classes overlap heavily.")
    else:
        print("RESULT: Some separation observed; check plots for details.")


if __name__ == "__main__":
    main()
