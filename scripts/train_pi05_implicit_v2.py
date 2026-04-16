#!/usr/bin/env python3
"""Deprecated wrapper for implicit V2 training.

Use the standard LeRobot training entrypoint with configs/train_pi05_implicit_v2.yaml.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lerobot.utils.import_utils import register_third_party_plugins

from scripts.pi05_implicit_policy import register_pi05_implicit_policy


def main():
    register_third_party_plugins()
    register_pi05_implicit_policy()
    import sys as _sys

    argv = list(_sys.argv)
    if not any(arg == '--config_path' or arg.startswith('--config_path=') for arg in argv):
        argv.extend(['--config_path', 'configs/train_pi05_implicit_v2.yaml'])
    _sys.argv = argv

    from lerobot.scripts.lerobot_train import train

    train()


if __name__ == "__main__":
    main()
