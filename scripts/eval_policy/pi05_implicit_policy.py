from __future__ import annotations

from typing import Optional

from lerobot.configs.policies import PreTrainedConfig

from lehome.utils.logger import get_logger
from scripts.pi05_implicit_policy import register_pi05_implicit_policy
from scripts.utils.garment_latent_utils import resolve_implicit_variant
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


@PolicyRegistry.register("pi05_implicit")
class Pi05ImplicitPolicy(LeRobotPolicy):
    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str,
        device: str = "cuda",
        use_delta_actions: bool = False,
        delta_stats_path: Optional[str] = None,
    ):
        register_pi05_implicit_policy()
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides={})
        policy_type = getattr(policy_cfg, "type", None) or getattr(policy_cfg.__class__, "type", None)
        if policy_type != "pi05_implicit":
            raise ValueError(
                f"Checkpoint at {policy_path} is not a pi05_implicit model (found type={policy_type!r})."
            )
        variant = resolve_implicit_variant(policy_cfg)
        logger.info(
            f"Pi05ImplicitPolicy adapter selected. config_type={policy_type}, implicit_variant={variant}"
        )
        super().__init__(
            policy_path=policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
            use_delta_actions=use_delta_actions,
            delta_stats_path=delta_stats_path,
        )
