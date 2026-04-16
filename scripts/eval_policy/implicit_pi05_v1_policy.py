from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("implicit_pi05_v1")
class ImplicitPi05V1Policy(BasePolicy):
    """Deprecated compatibility shim. Use policy_type='lerobot' for implicit checkpoints."""

    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str,
        device: str = "cuda",
        latent_dim: int = 8,
        use_delta_actions: bool = False,
        delta_stats_path: Optional[str] = None,
    ):
        del latent_dim
        self.policy = LeRobotPolicy(
            policy_path=policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
            use_delta_actions=use_delta_actions,
            delta_stats_path=delta_stats_path,
        )

    def reset(self):
        self.policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        return self.policy.select_action(observation)
