#!/usr/bin/env python3
"""Merge reject sampling data with original training set.

1. Scans reject_sampling/{garment_type}/{episode_id}/ directories
2. Filters out empty episodes (total_episodes == 0)
3. Merges valid episodes per garment type into merged datasets
4. Optionally merges everything (original + reject) into a final training set

Usage:
    # Step 1 & 2: Merge per garment type, then combine with original dataset
    python scripts/merge_reject_sampling.py \
        --reject_root Datasets/reject_sampling \
        --original_root Datasets/example/four_types_merged \
        --output_root Datasets/example/four_types_with_reject \
        --output_repo_id four_types_with_reject

    # Only merge reject sampling episodes per garment type (skip final combine)
    python scripts/merge_reject_sampling.py \
        --reject_root Datasets/reject_sampling \
        --skip_final_merge

    # Custom min_frames threshold (default 0, only filters total_episodes==0)
    python scripts/merge_reject_sampling.py \
        --reject_root Datasets/reject_sampling \
        --min_frames 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path for `scripts.utils` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.datasets.dataset_tools import merge_datasets as lerobot_merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripts.utils.dataset_processing import merge_garment_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_garment_types(reject_root: Path) -> list[str]:
    """Find garment type directories (exclude already merged ones)."""
    garment_types = []
    for p in sorted(reject_root.iterdir()):
        if p.is_dir() and not p.name.endswith("_merged"):
            garment_types.append(p.name)
    return garment_types


def scan_episodes(garment_dir: Path, min_frames: int = 0) -> list[Path]:
    """Scan episode directories and filter out empty ones.

    Args:
        garment_dir: Path to garment type directory (e.g. reject_sampling/pant_long)
        min_frames: Minimum total_frames threshold (0 = only filter total_episodes==0)

    Returns:
        List of valid episode directory paths
    """
    valid = []
    skipped = []
    for ep_dir in sorted(garment_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        info_path = ep_dir / "meta" / "info.json"
        if not info_path.exists():
            logger.warning(f"No info.json found in {ep_dir}, skipping")
            skipped.append((ep_dir.name, "no info.json"))
            continue

        info = json.loads(info_path.read_text())
        total_episodes = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)

        if total_episodes == 0 or total_frames < min_frames:
            reason = f"episodes={total_episodes}, frames={total_frames} < min_frames={min_frames}"
            logger.info(f"Skipping {ep_dir.name}: {reason}")
            skipped.append((ep_dir.name, reason))
        else:
            valid.append(ep_dir)
            logger.info(f"  Valid: {ep_dir.name} (episodes={total_episodes}, frames={total_frames})")

    logger.info(
        f"{garment_dir.name}: {len(valid)} valid, {len(skipped)} skipped"
    )
    return valid


def merge_per_garment(
    reject_root: Path,
    min_frames: int,
) -> dict[str, Path]:
    """Merge reject sampling episodes per garment type.

    Returns:
        Dict mapping garment_type -> merged dataset path
    """
    garment_types = find_garment_types(reject_root)
    merged_paths = {}

    for garment_type in garment_types:
        garment_dir = reject_root / garment_type
        output_dir = reject_root / f"{garment_type}_merged"

        if output_dir.exists():
            logger.info(f"{garment_type}_merged already exists, skipping")
            merged_paths[garment_type] = output_dir
            continue

        logger.info(f"=== Processing {garment_type} ===")
        valid_episodes = scan_episodes(garment_dir, min_frames)

        if not valid_episodes:
            logger.warning(f"No valid episodes for {garment_type}, skipping")
            continue

        # Merge episodes within this garment type
        datasets = []
        for ep_dir in valid_episodes:
            ds = LeRobotDataset(repo_id=ep_dir.name, root=ep_dir)
            datasets.append(ds)
            logger.info(f"Loaded {ep_dir.name}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")

        merged_ds = lerobot_merge_datasets(
            datasets=datasets,
            output_repo_id=f"{garment_type}_merged",
            output_dir=output_dir,
        )
        logger.info(
            f"Merged {garment_type}: {merged_ds.meta.total_episodes} episodes, "
            f"{merged_ds.meta.total_frames} frames -> {output_dir}"
        )

        # Merge garment_info.json
        merge_garment_info(valid_episodes, output_dir)
        logger.info(f"Garment info merged for {garment_type}")

        merged_paths[garment_type] = output_dir

    return merged_paths


def merge_final(
    original_root: Path,
    reject_merged_paths: dict[str, Path],
    output_root: Path,
    output_repo_id: str,
) -> None:
    """Merge original dataset with all reject sampling merged datasets."""
    source_roots = [original_root] + list(reject_merged_paths.values())

    logger.info("=== Final merge ===")
    logger.info(f"Source datasets ({len(source_roots)}):")
    for i, root in enumerate(source_roots, 1):
        logger.info(f"  {i}. {root}")

    datasets = []
    for root in source_roots:
        ds = LeRobotDataset(repo_id=root.name, root=root)
        datasets.append(ds)
        logger.info(f"Loaded {root.name}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")

    merged_ds = lerobot_merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_root,
    )
    logger.info(
        f"Final merged: {merged_ds.meta.total_episodes} episodes, "
        f"{merged_ds.meta.total_frames} frames -> {output_root}"
    )

    merge_garment_info(source_roots, output_root)
    logger.info("Final garment info merged")


def main():
    parser = argparse.ArgumentParser(
        description="Merge reject sampling data with original training set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reject_root",
        type=str,
        default="Datasets/reject_sampling",
        help="Root directory of reject sampling data (default: Datasets/reject_sampling)",
    )
    parser.add_argument(
        "--original_root",
        type=str,
        default="Datasets/example/four_types_merged",
        help="Path to original training dataset (default: Datasets/example/four_types_merged)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="Datasets/example/four_types_with_reject",
        help="Output path for final merged dataset (default: Datasets/example/four_types_with_reject)",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default="four_types_with_reject",
        help="Repository ID for final merged dataset (default: four_types_with_reject)",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=0,
        help="Minimum total_frames threshold per episode dir (default: 0, only filter total_episodes==0)",
    )
    parser.add_argument(
        "--skip_final_merge",
        action="store_true",
        help="Only merge reject sampling per garment type, skip combining with original dataset",
    )
    args = parser.parse_args()

    reject_root = Path(args.reject_root)
    original_root = Path(args.original_root)
    output_root = Path(args.output_root)

    # Step 1 & 2: Merge per garment type
    merged_paths = merge_per_garment(reject_root, args.min_frames)

    if not merged_paths:
        logger.error("No garment types merged, nothing to do")
        sys.exit(1)

    # Step 3: Combine with original dataset
    if not args.skip_final_merge:
        if not original_root.exists():
            logger.error(f"Original dataset not found: {original_root}")
            sys.exit(1)
        merge_final(original_root, merged_paths, output_root, args.output_repo_id)
    else:
        logger.info("Skipping final merge (--skip_final_merge)")
        for garment_type, path in merged_paths.items():
            logger.info(f"  {garment_type}: {path}")


if __name__ == "__main__":
    main()
