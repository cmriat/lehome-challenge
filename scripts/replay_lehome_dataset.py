"""
Standalone replay script - reads parquet directly, no LeRobot dependency.

Replays recorded episodes in Isaac Sim for visualization and verification.
Data is loaded directly from parquet files using pyarrow, avoiding the
heavy LeRobot dataset dependency.

  pixi run replay-standalone \
      --dataset_root Datasets/example/four_types_merged \
      --start_episode 0 \
      --end_episode 4 

"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for standalone replay.

    Only includes parameters needed for visualization replay.
    Saving/IK/depth parameters are intentionally excluded.
    """
    parser = argparse.ArgumentParser(
        description="Standalone replay - visualize recorded episodes in Isaac Sim (no LeRobot dependency)",
    )

    parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v2",
                        help="Isaac Lab task name")
    parser.add_argument("--step_hz", type=int, default=60,
                        help="Environment step frequency (Hz)")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Start episode index (inclusive)")
    parser.add_argument("--end_episode", type=int, default=None,
                        help="End episode index (exclusive)")
    parser.add_argument("--task_description", type=str, default="fold the garment on the table",
                        help="Task description string")
    parser.add_argument("--garment_version", type=str, default="Release",
                        help="Garment version")
    parser.add_argument("--garment_cfg_base_path", type=str,
                        default="Assets/objects/Challenge_Garment",
                        help="Base path for garment configurations")
    parser.add_argument("--particle_cfg_path", type=str,
                        default="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml",
                        help="Path to particle garment config")
    parser.add_argument("--num_replays", type=int, default=1,
                        help="Number of times to replay each episode")
    parser.add_argument("--save_video", action="store_true", default=False,
                        help="Save replay videos")
    parser.add_argument("--video_dir", type=str, default="outputs/replay_videos",
                        help="Directory to save replay videos")
    # NOTE: --device is provided by AppLauncher.add_app_launcher_args(), do not add here
    # Default is cuda:0 from AppLauncher, but this env's cloth simulation
    # requires CPU mode for correct physics behavior.

    return parser


# ---------------------------------------------------------------------------
# Data loading (pure Python / pyarrow, no Isaac dependency)
# ---------------------------------------------------------------------------

def load_dataset_info(dataset_root: str) -> Dict[str, Any]:
    """Load metadata from meta/info.json."""
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info.json not found: {info_path}")
    with open(info_path, "r") as f:
        return json.load(f)


def get_garment_name_from_json(dataset_root: str) -> str:
    """Parse garment_info.json to retrieve the garment name (top-level key).

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        The name of the garment (e.g., 'Top_Long_Seen_0').
    """
    pose_file = Path(dataset_root) / "meta" / "garment_info.json"
    if not pose_file.exists():
        raise FileNotFoundError(f"Garment info file not found: {pose_file}")

    with open(pose_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"Garment info file is empty: {pose_file}")

    garment_name = list(data.keys())[0]
    return garment_name


def load_initial_pose(
    dataset_root: str, episode_index: int
) -> Optional[Dict[str, Any]]:
    """Load the initial object pose for a given episode from garment_info.json.

    The pose file format is:
    {
      "Garment_Name": {
        "0": {
          "object_initial_pose": [x, y, z, roll, pitch, yaw],
          "scale": [...]
        }
      }
    }

    Args:
        dataset_root: Root directory of the dataset.
        episode_index: Index of the episode to load pose for.

    Returns:
        Dictionary {"Garment": [x, y, z, roll, pitch, yaw]} for env.set_all_pose(),
        or None if not found.
    """
    pose_file = Path(dataset_root) / "meta" / "garment_info.json"
    if not pose_file.exists():
        return None

    try:
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        episode_key = str(episode_index)
        for _garment_name, episodes in data.items():
            if episode_key in episodes:
                pose_list = episodes[episode_key].get("object_initial_pose")
                if pose_list is not None:
                    return {"Garment": pose_list}

        return None

    except (json.JSONDecodeError, KeyError):
        return None


def load_episode_actions(
    dataset_root: str, episode_index: int
) -> List[List[float]]:
    """Load action sequence for a specific episode from parquet files.

    Only reads 'episode_index' and 'action' columns to avoid loading
    image data into memory.

    Args:
        dataset_root: Root directory of the dataset.
        episode_index: The episode index to load.

    Returns:
        List of action vectors, each a list of floats.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    data_root = Path(dataset_root) / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_root}")

    all_actions: List[List[float]] = []
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["episode_index", "action"])
        ep_indices = table["episode_index"].to_pylist()
        mask = [i for i, ep in enumerate(ep_indices) if ep == episode_index]
        if mask:
            episode_table = table.take(mask)
            all_actions.extend(episode_table["action"].to_pylist())

    return all_actions


# ---------------------------------------------------------------------------
# Replay logic (depends on Isaac Sim - imported inside main)
# ---------------------------------------------------------------------------

def replay_episode(
    env,
    actions: List[List[float]],
    initial_pose: Optional[Dict[str, Any]],
    rate_limiter,
    stabilize_fn,
    args: argparse.Namespace,
    device: str,
    save_video: bool = False,
) -> Tuple[bool, Dict[str, List[np.ndarray]]]:
    """Replay a single episode from its action sequence.

    Args:
        env: Isaac Lab environment instance (DirectRLEnv).
        actions: List of action vectors for this episode.
        initial_pose: Initial garment pose dict, or None.
        rate_limiter: RateLimiter instance, or None.
        stabilize_fn: stabilize_garment_after_reset function.
        device: Compute device string.
        save_video: Whether to collect frames for video saving.

    Returns:
        Tuple of (success_achieved, frames_dict).
        frames_dict maps camera key -> list of HWC uint8 RGB frames.
        Empty dict when save_video is False.
    """
    env.reset()

    if initial_pose is not None:
        env.set_all_pose(initial_pose)

    stabilize_fn(env, args)

    success_achieved = False
    frames: Dict[str, List[np.ndarray]] = {}

    for action_list in tqdm(actions, desc="  Replaying", unit="frame"):
        if rate_limiter:
            rate_limiter.sleep(env)

        action = torch.tensor(action_list, dtype=torch.float32, device=device).unsqueeze(0)
        env.step(action)

        if env._get_success().item():
            success_achieved = True

        if save_video:
            obs = env._get_observations()
            for key, value in obs.items():
                if "images" in key:
                    frames.setdefault(key, []).append(value.copy())

    return success_achieved, frames


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # 1. Build parser and add Isaac Sim launcher arguments
    parser = build_parser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Force CPU device: this environment's cloth/particle simulation requires
    # CPU mode for correct physics behavior. GPU mode causes the simulation
    # to appear frozen (arms and garment do not move).
    if not any(arg.startswith("--device") for arg in __import__("sys").argv[1:]):
        args.device = "cpu"

    # 2. Launch Isaac Sim application
    from scripts.utils.common import launch_app_from_args, close_app
    simulation_app = launch_app_from_args(args)

    try:
        # 3. Register task (must happen AFTER SimulationApp is created)
        import lehome.tasks.bedroom

        # 4. Import Isaac-dependent modules
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        from lehome.utils.record import RateLimiter
        from lehome.utils.logger import get_logger
        from scripts.utils.common import stabilize_garment_after_reset
        from scripts.utils.eval_utils import save_videos_from_observations

        logger = get_logger(__name__)

        # 5. Validate dataset
        dataset_path = Path(args.dataset_root)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {args.dataset_root}")

        info = load_dataset_info(args.dataset_root)
        total_episodes = info.get("total_episodes", 0)
        if total_episodes == 0:
            raise ValueError("Dataset has 0 episodes")

        # 6. Determine episode range
        start_idx = args.start_episode
        end_idx = args.end_episode if args.end_episode is not None else total_episodes
        end_idx = min(end_idx, total_episodes)

        if start_idx < 0:
            raise ValueError(f"start_episode must be >= 0, got {start_idx}")
        if end_idx <= start_idx:
            raise ValueError(
                f"end_episode ({end_idx}) must be > start_episode ({start_idx})"
            )

        # 7. Detect garment name from dataset
        garment_name = get_garment_name_from_json(args.dataset_root)
        logger.info(f"Detected garment: {garment_name}")

        # 8. Setup environment
        device = args.device
        env_cfg = parse_env_cfg(args.task, device=device)
        env_cfg.garment_name = garment_name
        env_cfg.garment_version = args.garment_version
        env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
        env_cfg.particle_cfg_path = args.particle_cfg_path

        env = gym.make(args.task, cfg=env_cfg).unwrapped

        # Initialize observations (required for garment initialization)
        logger.info("Initializing environment...")
        env.initialize_obs()
        logger.info("Environment initialized")

        # 9. Create rate limiter
        rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None

        # 10. Replay loop
        total_count = end_idx - start_idx
        total_attempts = 0
        total_successes = 0

        logger.info(f"Replaying episodes {start_idx} to {end_idx - 1} "
                     f"(total {total_count}, {args.num_replays} replay(s) each)")

        for ep_idx in range(start_idx, end_idx):
            display_num = ep_idx - start_idx + 1

            logger.info("")
            logger.info(f"{'=' * 60}")
            logger.info(f"Episode {display_num}/{total_count} (index={ep_idx})")
            logger.info(f"{'=' * 60}")

            # Load initial pose and actions
            initial_pose = load_initial_pose(args.dataset_root, ep_idx)
            actions = load_episode_actions(args.dataset_root, ep_idx)

            if not actions:
                logger.warning(f"No actions found for episode {ep_idx}, skipping")
                continue

            logger.info(f"  Frames: {len(actions)}")

            for replay_idx in range(args.num_replays):
                total_attempts += 1

                success, frames = replay_episode(
                    env=env,
                    actions=actions,
                    initial_pose=initial_pose,
                    rate_limiter=rate_limiter,
                    stabilize_fn=stabilize_garment_after_reset,
                    args=args,
                    device=device,
                    save_video=args.save_video,
                )

                if success:
                    total_successes += 1

                if args.save_video and frames:
                    success_tensor = torch.tensor(success)
                    save_videos_from_observations(
                        frames, args.video_dir, ep_idx, success_tensor,
                    )

                if args.num_replays > 1:
                    status = "Success" if success else "Failed"
                    logger.info(f"  [Replay {replay_idx + 1}/{args.num_replays}] {status}")
                else:
                    logger.info(f"  Result: {'Success' if success else 'Failed'}")

        # 11. Print summary
        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info("Replay Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Total attempts: {total_attempts}")
        logger.info(f"  Successes:      {total_successes}")
        if total_attempts > 0:
            logger.info(f"  Success rate:   {100.0 * total_successes / total_attempts:.1f}%")
        logger.info(f"{'=' * 60}")

        env.close()

    except Exception:
        from lehome.utils.logger import get_logger
        get_logger(__name__).error("Replay failed", exc_info=True)
        raise

    finally:
        close_app(simulation_app)


if __name__ == "__main__":
    main()
