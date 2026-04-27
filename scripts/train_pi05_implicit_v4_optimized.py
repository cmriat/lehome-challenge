#!/usr/bin/env python3
"""Optimized implicit V4 training entrypoint for Pi0.5 on H20.

Applies training optimizations (+persistent_workers, +TF32 matmul),
registers pi05_implicit policy, then delegates to lerobot-train.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lerobot.utils.import_utils import register_third_party_plugins
from scripts.pi05_implicit_policy import register_pi05_implicit_policy
from scripts.train_utils.training_optimizations import apply_all_optimizations


def main():
    register_third_party_plugins()
    register_pi05_implicit_policy()
    apply_all_optimizations()

    import sys as _sys

    argv = list(_sys.argv)
    if not any(arg == "--config_path" or arg.startswith("--config_path=") for arg in argv):
        argv.extend(["--config_path", "configs/train_pi05_implicit_v4_final_3k.yaml"])
    _sys.argv = argv

    from lerobot.scripts.lerobot_train import train

    train()


if __name__ == "__main__":
    main()
