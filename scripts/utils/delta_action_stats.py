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
        actions = np.array(table["action"].to_pylist(), dtype=np.float32)       # (N, action_dim)
        states = np.array(table["observation.state"].to_pylist(), dtype=np.float32)  # (N, action_dim)
        episode_indices = table["episode_index"].to_pylist()

        # Group frame indices by episode
        episodes = {}
        for idx, ep_idx in enumerate(episode_indices):
            if ep_idx not in episodes:
                episodes[ep_idx] = []
            episodes[ep_idx].append(idx)

        for ep_idx in sorted(episodes.keys()):
            frame_indices = episodes[ep_idx]
            ep_start = min(frame_indices)
            ep_end = max(frame_indices) + 1

            # For each valid starting frame t0
            for t0 in range(ep_start, ep_end):
                chunk_end = min(t0 + chunk_size, ep_end)
                actual_chunk_len = chunk_end - t0

                # Get state at t0
                state_t0 = states[t0]  # (action_dim,)

                # Get action chunk [t0, t0+chunk_size), clamped to episode boundary
                # If chunk extends beyond episode, the last valid action is repeated
                # (matching LeRobot's _get_query_indices behavior)
                chunk_actions = np.zeros((chunk_size, action_dim), dtype=np.float32)
                chunk_actions[:actual_chunk_len] = actions[t0:chunk_end]
                if actual_chunk_len < chunk_size:
                    # Repeat last valid action for padding (LeRobot clamping behavior)
                    chunk_actions[actual_chunk_len:] = actions[chunk_end - 1]

                # Compute delta: action[t0+i] - state[t0]
                delta_chunk = chunk_actions - state_t0[np.newaxis, :]  # (chunk_size, action_dim)

                # Feed into running stats
                running_stats.update(delta_chunk)
                total_chunks += 1

        logger.info(f"  Processed {pf.name}: {len(episodes)} episodes, {total_chunks} chunks so far")

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
