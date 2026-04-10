"""Compute delta action statistics for per-chunk delta training.

For per-chunk delta, each action in the chunk is relative to the same observation.state[t0]:
    delta_action[t0+i] = action[t0+i] - observation.state[t0]  (i = 0..chunk_size-1)

This script simulates the actual delta distribution seen during training and computes
quantile statistics (q01, q10, q50, q90, q99) needed for QUANTILE normalization.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.compute_stats import RunningQuantileStats

from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def compute_delta_action_stats(
    dataset_root: Path,
    chunk_size: int = 30,
    output_path: Optional[Path] = None,
) -> dict:
    """Compute per-chunk delta action statistics from a LeRobot v3.0 dataset.

    For each valid starting frame t0 within an episode, compute:
        delta[i] = action[t0+i] - observation.state[t0]  for i in [0, chunk_size)

    Accumulate all delta values into RunningQuantileStats.

    Args:
        dataset_root: Root directory of the LeRobot dataset.
        chunk_size: Action chunk size (must match training config).
        output_path: Where to save the stats JSON. Defaults to dataset_root/meta/delta_action_stats.json.

    Returns:
        Dictionary of stats: {min, max, mean, std, count, q01, q10, q50, q90, q99}.
        Each value is a list of floats (length = action_dim).
    """
    dataset_root = Path(dataset_root).resolve()
    if output_path is None:
        output_path = dataset_root / "meta" / "delta_action_stats.json"

    # Load info.json for action dim
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    action_dim = info["features"]["action"]["shape"][0]
    total_episodes = info["total_episodes"]

    logger.info(
        f"Computing per-chunk delta action stats: "
        f"chunk_size={chunk_size}, action_dim={action_dim}, episodes={total_episodes}"
    )

    # Read all data parquet files
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    running_stats = RunningQuantileStats()
    total_chunks = 0

    for pf in parquet_files:
        table = pq.read_table(pf)

        # Fast Arrow → numpy: flatten list columns directly (avoids slow to_pylist())
        actions = table["action"].combine_chunks().flatten().to_numpy(
            zero_copy_only=False
        ).reshape(-1, action_dim).astype(np.float32)
        states = table["observation.state"].combine_chunks().flatten().to_numpy(
            zero_copy_only=False
        ).reshape(-1, action_dim).astype(np.float32)
        episode_indices = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False)

        # Find episode boundaries using diff (assumes rows are sorted by episode)
        ep_change = np.diff(episode_indices, prepend=episode_indices[0] - 1)
        ep_starts = np.where(ep_change != 0)[0]
        ep_lengths = np.diff(np.append(ep_starts, len(episode_indices)))

        all_deltas = []
        for i in tqdm(range(len(ep_starts)), desc=f"Processing {pf.name}", unit="ep"):
            ep_start = ep_starts[i]
            ep_len = ep_lengths[i]

            # Build clamped index array: shape (ep_len, chunk_size)
            # indices[t, j] = min(t + j, ep_len - 1), offset by ep_start
            offsets = np.arange(chunk_size)  # (chunk_size,)
            t0s = np.arange(ep_len)[:, np.newaxis]  # (ep_len, 1)
            indices = np.minimum(t0s + offsets, ep_len - 1) + ep_start  # (ep_len, chunk_size)

            # Gather actions and states via vectorized indexing
            chunk_actions = actions[indices]  # (ep_len, chunk_size, action_dim)
            state_t0 = states[ep_start:ep_start + ep_len, np.newaxis, :]  # (ep_len, 1, action_dim)

            # Vectorized delta computation for entire episode
            deltas = chunk_actions - state_t0  # (ep_len, chunk_size, action_dim)
            all_deltas.append(deltas.reshape(-1, action_dim))
            total_chunks += ep_len

        # Batch update: single call per parquet file
        if all_deltas:
            all_deltas = np.concatenate(all_deltas, axis=0)  # (total_frames * chunk_size, action_dim)
            running_stats.update(all_deltas)

        num_episodes = len(ep_starts)
        logger.info(f"  Processed {pf.name}: {num_episodes} episodes, {total_chunks} chunks so far")

    # Get final stats
    stats = running_stats.get_statistics()
    logger.info(
        f"Done. Total chunks: {total_chunks}, "
        f"total delta samples: {total_chunks * chunk_size}"
    )

    # Log some stats for sanity check
    logger.info(f"  Delta action mean: {stats['mean']}")
    logger.info(f"  Delta action std:  {stats['std']}")
    logger.info(f"  Delta action q01:  {stats['q01']}")
    logger.info(f"  Delta action q99:  {stats['q99']}")

    # Convert to JSON-serializable format (list of floats)
    stats_json = {}
    for key, value in stats.items():
        stats_json[key] = value.tolist()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(stats_json, f, indent=2)
    logger.info(f"Saved delta action stats to {output_path}")

    return stats_json


def load_delta_action_stats(stats_path: Path) -> dict:
    """Load delta action stats and convert to numpy arrays.

    Args:
        stats_path: Path to the delta_action_stats.json file.

    Returns:
        Dictionary with numpy array values matching stats.json format for "action".
    """
    with open(stats_path, "r") as f:
        stats_json = json.load(f)

    stats = {}
    for key, value in stats_json.items():
        stats[key] = np.array(value, dtype=np.float32)

    return stats
