#!/usr/bin/env python

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lerobot.utils.import_utils import register_third_party_plugins


def main():
    register_third_party_plugins()
    from lerobot.scripts.lerobot_train import train

    train()


if __name__ == "__main__":
    main()
