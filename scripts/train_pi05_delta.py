#!/usr/bin/env python
"""
Delta action training wrapper for pi0.5.

This script wraps the standard lerobot-train entrypoint to inject per-chunk
delta action conversion into the preprocessor pipeline. It:

1. Patches `make_pre_post_processors` to inject DeltaActionProcessorStep
   and replace action normalization stats with delta stats.
2. Delegates everything else to the standard lerobot train() function.

Usage (same as lerobot-train, just use this script instead):
    accelerate launch --mixed_precision=bf16 --num_processes=8 \
        scripts/train_pi05_delta.py \
        --config_path=configs/train_pi05_delta.yaml \
        2>&1 | tee logs/train_pi05_delta_$(date +%Y%m%d_%H%M%S).log
"""

import json
import logging
import sys
from functools import wraps
from pathlib import Path

import numpy as np

from lerobot.utils.import_utils import register_third_party_plugins


logger = logging.getLogger(__name__)


def _find_delta_stats_path(cfg) -> Path:
    """Locate delta_action_stats.json from the dataset config.

    Searches in order:
    1. cfg.delta_stats_path (if the user added a custom field)
    2. <dataset_root>/meta/delta_action_stats.json
    """
    # Try custom config field
    delta_path = getattr(cfg, "delta_stats_path", None)
    if delta_path and Path(delta_path).exists():
        return Path(delta_path)

    # Try dataset root
    dataset_root = Path(cfg.dataset.root)
    stats_path = dataset_root / "meta" / "delta_action_stats.json"
    if stats_path.exists():
        return stats_path

    raise FileNotFoundError(
        f"Delta action stats not found. Searched:\n"
        f"  1. cfg.delta_stats_path = {delta_path}\n"
        f"  2. {stats_path}\n"
        f"Run `python -m scripts.dataset delta-stats --dataset_root <path> --chunk_size 30` first."
    )


def _load_delta_stats(path: Path) -> dict:
    """Load delta action stats JSON and convert values to numpy arrays."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def _patch_make_pre_post_processors(delta_stats: dict):
    """Monkey-patch make_pre_post_processors to inject delta action support.

    The patch:
    1. Calls the original make_pre_post_processors
    2. Inserts DeltaActionProcessorStep before NormalizerProcessorStep
    3. Replaces action stats in both preprocessor (normalizer) and postprocessor (unnormalizer)
    """
    import lerobot.scripts.lerobot_train as train_module
    from lerobot.policies.factory import make_pre_post_processors as original_make

    from scripts.utils.delta_action_processor import (
        inject_delta_step_into_preprocessor,
        replace_action_stats_in_processor,
    )

    @wraps(original_make)
    def patched_make(*args, **kwargs):
        # Replace stats in the overrides before creating processors
        # This ensures the NormalizerProcessorStep is initialized with delta stats
        if "preprocessor_overrides" in kwargs:
            overrides = kwargs["preprocessor_overrides"]
            if "normalizer_processor" in overrides and "stats" in overrides["normalizer_processor"]:
                stats = overrides["normalizer_processor"]["stats"]
                stats["action"] = delta_stats
                logger.info("Replaced action stats in preprocessor_overrides with delta stats")

        if "postprocessor_overrides" in kwargs:
            overrides = kwargs["postprocessor_overrides"]
            if "unnormalizer_processor" in overrides and "stats" in overrides["unnormalizer_processor"]:
                stats = overrides["unnormalizer_processor"]["stats"]
                stats["action"] = delta_stats
                logger.info("Replaced action stats in postprocessor_overrides with delta stats")

        # Also handle dataset_stats kwarg (used when no pretrained_path)
        if "dataset_stats" in kwargs and kwargs["dataset_stats"] is not None:
            kwargs["dataset_stats"]["action"] = delta_stats
            logger.info("Replaced action stats in dataset_stats with delta stats")

        preprocessor, postprocessor = original_make(*args, **kwargs)

        # Inject DeltaActionProcessorStep into preprocessor
        inject_delta_step_into_preprocessor(preprocessor)
        logger.info("Injected DeltaActionProcessorStep into preprocessor pipeline")

        # Also replace stats in the actual processor steps (in case they were loaded from checkpoint)
        replace_action_stats_in_processor(preprocessor, delta_stats)
        replace_action_stats_in_processor(postprocessor, delta_stats)

        return preprocessor, postprocessor

    # Patch both the factory module and the train module's imported reference
    import lerobot.policies.factory as factory_module
    factory_module.make_pre_post_processors = patched_make
    train_module.make_pre_post_processors = patched_make


def main():
    """Entry point: patch processors, then run standard lerobot training."""
    register_third_party_plugins()

    # We need to parse the config first to find delta stats path.
    # Use a lightweight pre-parse to extract dataset.root from CLI args.
    delta_stats_path = None
    dataset_root = None

    # Extract --config_path and dataset.root from CLI args
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--config_path="):
            config_path = arg.split("=", 1)[1]
        elif arg == "--config_path" and i + 1 < len(sys.argv[1:]):
            config_path = sys.argv[i + 2]

    # Try to parse config YAML for dataset root
    try:
        import yaml
        config_path_val = None
        for i, arg in enumerate(sys.argv):
            if "config_path" in arg:
                if "=" in arg:
                    config_path_val = arg.split("=", 1)[1]
                elif i + 1 < len(sys.argv):
                    config_path_val = sys.argv[i + 1]
                break

        if config_path_val:
            with open(config_path_val, "r") as f:
                yaml_cfg = yaml.safe_load(f)
            dataset_root = yaml_cfg.get("dataset", {}).get("root")
            # Check for custom delta_stats_path in yaml
            delta_stats_path_str = yaml_cfg.get("delta_stats_path")
            if delta_stats_path_str:
                delta_stats_path = Path(delta_stats_path_str)
    except Exception as e:
        logger.warning(f"Failed to pre-parse config: {e}")

    # Find delta stats
    if delta_stats_path is None and dataset_root:
        candidate = Path(dataset_root) / "meta" / "delta_action_stats.json"
        if candidate.exists():
            delta_stats_path = candidate

    if delta_stats_path is None or not delta_stats_path.exists():
        print(
            "ERROR: Cannot find delta_action_stats.json.\n"
            "Run `python -m scripts.dataset delta-stats --dataset_root <path> --chunk_size 30` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    delta_stats = _load_delta_stats(delta_stats_path)
    logger.info(f"Loaded delta action stats from {delta_stats_path}")
    logger.info(f"  Delta q01: {delta_stats.get('q01', 'N/A')}")
    logger.info(f"  Delta q99: {delta_stats.get('q99', 'N/A')}")

    # Apply the monkey-patch
    _patch_make_pre_post_processors(delta_stats)

    # Now run the standard training function
    from lerobot.scripts.lerobot_train import train
    train()


if __name__ == "__main__":
    main()
