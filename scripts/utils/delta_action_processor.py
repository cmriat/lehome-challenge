"""Delta action processor step for per-chunk delta training.

Converts absolute actions to per-chunk delta actions in the preprocessor pipeline:
    delta_action[i] = action[i] - observation.state  (broadcast over chunk dimension)

This step should be inserted AFTER AddBatchDimensionProcessorStep and BEFORE
NormalizerProcessorStep in the pi0.5 preprocessor pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep


OBS_STATE_KEY = "observation.state"


@dataclass
class DeltaActionProcessorStep(ProcessorStep):
    """Converts absolute actions to per-chunk delta actions.

    For per-chunk delta, all actions in the chunk are relative to the
    same observation.state (the current state at chunk start):
        delta = action - state.unsqueeze(-2)

    Handles both training batches [B, chunk_size, action_dim] and
    inference single samples [1, action_dim] or [1, chunk_size, action_dim].
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Apply delta conversion to action in the transition.

        Args:
            transition: EnvTransition containing observation and action.

        Returns:
            Modified transition with delta actions.
        """
        new_transition = deepcopy(transition)

        # Get observation state
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE_KEY not in observation:
            # No observation state available, skip delta conversion
            return new_transition

        state = observation[OBS_STATE_KEY]  # [B, state_dim] or [state_dim]

        # Get action
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        # Handle PolicyAction (torch.Tensor)
        if isinstance(action, torch.Tensor):
            action_tensor = action
        else:
            return new_transition

        # Compute delta action:
        # state shape:  [B, state_dim] or [state_dim]
        # action shape: [B, chunk_size, action_dim] or [B, action_dim] or [action_dim]
        if action_tensor.ndim >= 2 and state.ndim >= 1:
            if action_tensor.ndim == state.ndim + 1:
                # Training case: action [B, chunk, dim], state [B, dim]
                # -> state.unsqueeze(-2) = [B, 1, dim], broadcast over chunk
                delta_action = action_tensor - state.unsqueeze(-2)
            elif action_tensor.ndim == state.ndim:
                # Inference case: action [B, dim], state [B, dim]
                delta_action = action_tensor - state
            else:
                # Fallback: try broadcasting
                delta_action = action_tensor - state.unsqueeze(-2)
        else:
            # 1D case or unexpected shapes
            delta_action = action_tensor - state

        new_transition[TransitionKey.ACTION] = delta_action

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Feature shapes don't change — delta has same shape as absolute action."""
        return features


def inject_delta_step_into_preprocessor(preprocessor: Any) -> None:
    """Insert DeltaActionProcessorStep into an existing preprocessor pipeline.

    Inserts after AddBatchDimensionProcessorStep and before NormalizerProcessorStep.

    Args:
        preprocessor: DataProcessorPipeline instance (pi05 preprocessor).
    """
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    delta_step = DeltaActionProcessorStep()

    # Find insertion point: before NormalizerProcessorStep
    steps = list(preprocessor.steps)
    insert_idx = None
    for i, step in enumerate(steps):
        if isinstance(step, NormalizerProcessorStep):
            insert_idx = i
            break

    if insert_idx is None:
        raise RuntimeError(
            "Could not find NormalizerProcessorStep in preprocessor pipeline. "
            "Cannot inject DeltaActionProcessorStep."
        )

    steps.insert(insert_idx, delta_step)
    preprocessor.steps = steps


def replace_action_stats_in_processor(
    processor: Any,
    delta_stats: dict,
) -> None:
    """Replace action normalization stats in a NormalizerProcessorStep or UnnormalizerProcessorStep.

    The normalizer uses two dicts:
    - self.stats["action"] = {"q01": np.array, "q99": np.array, ...}  (numpy)
    - self._tensor_stats["action"] = {"q01": Tensor, "q99": Tensor, ...}  (torch)

    Both must be replaced for the delta stats to take effect.

    Args:
        processor: DataProcessorPipeline containing a normalizer/unnormalizer step.
        delta_stats: Dictionary of delta action stats (numpy arrays),
            e.g. {"q01": np.array([...]), "q99": np.array([...]), "mean": ..., ...}.
    """
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    import torch
    import numpy as np

    for step in processor.steps:
        if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
            # Replace numpy stats
            if step.stats is not None and "action" in step.stats:
                step.stats["action"] = {
                    k: np.array(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
                    for k, v in delta_stats.items()
                }

            # Replace tensor stats
            if hasattr(step, "_tensor_stats") and "action" in step._tensor_stats:
                device = next(iter(step._tensor_stats["action"].values())).device
                dtype = next(iter(step._tensor_stats["action"].values())).dtype
                step._tensor_stats["action"] = {
                    k: torch.tensor(v, device=device, dtype=dtype)
                    if not isinstance(v, torch.Tensor) else v.to(device=device, dtype=dtype)
                    for k, v in delta_stats.items()
                }


def build_delta_stats_override(
    original_stats: dict,
    delta_stats: dict,
) -> dict:
    """Build a combined stats dict where 'action' stats are replaced with delta stats.

    This is used for preprocessor_overrides / postprocessor_overrides when constructing
    the training pipeline.

    Args:
        original_stats: The full dataset stats dict (from dataset.meta.stats).
        delta_stats: Delta action stats dict (numpy arrays).

    Returns:
        A new stats dict with action stats replaced.
    """
    import copy
    combined = copy.deepcopy(original_stats)
    combined["action"] = delta_stats
    return combined
