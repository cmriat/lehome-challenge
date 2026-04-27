"""
Training optimizations for Pi0.5 on H20 140G + PyTorch 2.7.

Apply before calling lerobot-train to squeeze max performance:
  1. torch.set_float32_matmul_precision('high') — TF32 acceleration
  2. persistent_workers=True patch — avoid DataLoader worker respawn
  3. NCCL env tuning hints (set in shell launcher)

Usage:
    from scripts.train_utils.training_optimizations import apply_all_optimizations
    apply_all_optimizations()
"""
from __future__ import annotations

import logging
from functools import wraps

import torch

logger = logging.getLogger(__name__)


def set_matmul_precision() -> None:
    """Enable TF32 tensor core acceleration (PyTorch >= 2.0).

    On H20 (SM90), tf32 is emulated via bf16 but still ~2x faster than fp32 matmul
    with negligible accuracy loss.
    """
    torch.set_float32_matmul_precision("high")
    logger.info("Set float32 matmul precision: high")


def _patch_dataloader_persistent_workers() -> None:
    """Monkey-patch DataLoader creation to enable persistent_workers.

    lerobot creates DataLoader without persistent_workers=True, causing
    worker processes to be destroyed and recreated each epoch. This costs
    ~2-5s per epoch in worker startup overhead.
    """
    from torch.utils.data import DataLoader as _OriginalDataLoader

    _original_init = _OriginalDataLoader.__init__

    @wraps(_original_init)
    def _patched_init(self, *args, **kwargs):
        if kwargs.get("num_workers", 0) > 0:
            kwargs.setdefault("persistent_workers", True)
        _original_init(self, *args, **kwargs)

    _OriginalDataLoader.__init__ = _patched_init
    logger.info("Patched DataLoader: persistent_workers=True")


def apply_all_optimizations() -> None:
    set_matmul_precision()
    _patch_dataloader_persistent_workers()
