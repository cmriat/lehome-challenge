from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.factory import _get_policy_cls_from_policy_name, _make_processors_from_policy_config
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from scripts.utils.garment_latent_utils import ImplicitConditioningConfig, ImplicitConditioner


@PreTrainedConfig.register_subclass("pi05_implicit")
@dataclass
class PI05ImplicitConfig(PI05Config):
    implicit_conditioning: dict = field(default_factory=dict)
    aux_type_loss_weight: float = 0.0
    aux_type_warmup_start_ratio: float = 0.1
    aux_type_warmup_end_ratio: float = 0.3
    total_training_steps: int = 0
    type_target_key: str = "observation.garment_type_prob"
    type_num_classes: int = 4
    split_latent: bool = False
    z_task_dim: int = 8
    z_type_dim: int = 4
    weak_coupling: bool = False
    weak_coupling_apply_at_inference: bool = True
    weak_coupling_gate_bias_init: float = 2.0
    weak_coupling_residual: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.implicit_conditioning:
            if isinstance(self.implicit_conditioning, ImplicitConditioningConfig):
                self.implicit_conditioning = asdict(self.implicit_conditioning)
            elif not isinstance(self.implicit_conditioning, dict):
                raise ValueError("implicit_conditioning must be a dict or ImplicitConditioningConfig")
        else:
            self.implicit_conditioning = {}

        latent_dim = self.implicit_conditioning.get("latent_dim", 0)
        if self.split_latent and latent_dim != self.z_task_dim + self.z_type_dim:
            raise ValueError(
                f"split latent mismatch: latent_dim={latent_dim}, z_task_dim+z_type_dim={self.z_task_dim + self.z_type_dim}"
            )

        if self.z_task_dim <= 0:
            raise ValueError("z_task_dim must be positive")

        if self.z_type_dim <= 0:
            raise ValueError("z_type_dim must be positive")

        if self.weak_coupling and not self.split_latent:
            raise ValueError("weak_coupling requires split_latent=True")

        if self.weak_coupling_gate_bias_init <= 0:
            raise ValueError("weak_coupling_gate_bias_init must be positive")

        if self.aux_type_loss_weight < 0:
            raise ValueError("aux_type_loss_weight must be non-negative")

        if self.total_training_steps < 0:
            raise ValueError("total_training_steps must be non-negative")

        if not 0 <= self.aux_type_warmup_start_ratio <= 1:
            raise ValueError("aux_type_warmup_start_ratio must be in [0,1]")

        if not 0 <= self.aux_type_warmup_end_ratio <= 1:
            raise ValueError("aux_type_warmup_end_ratio must be in [0,1]")

        if self.aux_type_warmup_end_ratio < self.aux_type_warmup_start_ratio:
            raise ValueError("aux_type_warmup_end_ratio must be >= aux_type_warmup_start_ratio")


class PI05ImplicitPolicy(PI05Policy):
    def _save_pretrained(self, save_directory):
        super()._save_pretrained(save_directory)
        torch.save({'train_step': self._train_step}, str(save_directory / 'implicit_aux_state.pt'))

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, config=None, **kwargs):
        model = super().from_pretrained(pretrained_name_or_path, *args, config=config, **kwargs)
        import os
        aux_path = os.path.join(str(pretrained_name_or_path), 'implicit_aux_state.pt')
        if os.path.exists(aux_path):
            state = torch.load(aux_path, map_location='cpu')
            model._train_step = int(state.get('train_step', 0))
        return model

    config_class = PI05ImplicitConfig
    name = "pi05_implicit"

    def __init__(self, config: PI05ImplicitConfig, **kwargs):
        self.implicit_conditioning = ImplicitConditioningConfig(**(config.implicit_conditioning or {}))
        self.aux_type_loss_weight = config.aux_type_loss_weight
        self.aux_type_warmup_start_ratio = config.aux_type_warmup_start_ratio
        self.aux_type_warmup_end_ratio = config.aux_type_warmup_end_ratio
        self.total_training_steps = config.total_training_steps
        self.type_target_key = config.type_target_key
        self.split_latent = config.split_latent
        self.z_task_dim = config.z_task_dim
        self.z_type_dim = config.z_type_dim
        self.weak_coupling = getattr(config, 'weak_coupling', True)
        self.weak_coupling_apply_at_inference = config.weak_coupling_apply_at_inference
        self.weak_coupling_gate_bias_init = config.weak_coupling_gate_bias_init
        self.weak_coupling_residual = config.weak_coupling_residual
        self._train_step = 0
        super().__init__(config, **kwargs)
        self.implicit_conditioner = ImplicitConditioner(self.implicit_conditioning)
        self.type_head = nn.Linear(self.z_type_dim if self.split_latent else self.implicit_conditioning.latent_dim, config.type_num_classes)
        self.task_gate_head = nn.Linear(self.z_type_dim, self.z_task_dim)
        self.task_bias_head = nn.Linear(self.z_type_dim, self.z_task_dim)
        nn.init.zeros_(self.task_gate_head.weight)
        nn.init.constant_(self.task_gate_head.bias, self.weak_coupling_gate_bias_init)
        nn.init.zeros_(self.task_bias_head.weight)
        nn.init.zeros_(self.task_bias_head.bias)

    def _split_latent(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        if not self.split_latent:
            return latent, latent
        return latent[:, : self.z_task_dim], latent[:, self.z_task_dim :]

    def _current_aux_weight(self) -> float:
        if self.aux_type_loss_weight <= 0 or self.total_training_steps <= 0:
            return self.aux_type_loss_weight
        progress = self._train_step / max(self.total_training_steps, 1)
        if progress <= self.aux_type_warmup_start_ratio:
            return 0.0
        if progress >= self.aux_type_warmup_end_ratio:
            return self.aux_type_loss_weight
        span = self.aux_type_warmup_end_ratio - self.aux_type_warmup_start_ratio
        alpha = (progress - self.aux_type_warmup_start_ratio) / max(span, 1e-8)
        return self.aux_type_loss_weight * alpha

    def _apply_implicit_conditioning(
        self,
        batch: dict[str, Tensor],
        *,
        apply_weak_coupling: bool,
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        if not self.implicit_conditioning.enabled:
            return batch, None
        source_key = self.implicit_conditioning.source_image_key
        target_key = self.implicit_conditioning.target_state_key
        if source_key not in batch or target_key not in batch:
            raise KeyError(f"Implicit conditioning requires '{source_key}' and '{target_key}'")
        image = batch[source_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        latent = self.implicit_conditioner.encode({source_key: image})
        z_task, z_type = self._split_latent(latent)
        if self.split_latent and self.weak_coupling and apply_weak_coupling:
            gate = torch.sigmoid(self.task_gate_head(z_type))
            bias = self.task_bias_head(z_type)
            weak_coupled = z_task * gate + bias
            z_task = weak_coupled + z_task if self.weak_coupling_residual else weak_coupled
        conditioned = dict(batch)
        conditioned[target_key] = torch.cat([batch[target_key], z_task], dim=-1)
        conditioned["_implicit_z_type"] = z_type
        return conditioned, z_type

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        conditioned, _ = self._apply_implicit_conditioning(
            batch,
            apply_weak_coupling=self.weak_coupling_apply_at_inference,
        )
        return super().select_action(conditioned, **kwargs)

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        conditioned, _ = self._apply_implicit_conditioning(
            batch,
            apply_weak_coupling=self.weak_coupling_apply_at_inference,
        )
        return super().predict_action_chunk(conditioned, **kwargs)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        conditioned, z_type = self._apply_implicit_conditioning(batch, apply_weak_coupling=True)
        policy_loss, loss_dict = super().forward(conditioned, reduction=reduction)
        if z_type is None or self.type_target_key not in batch or self.aux_type_loss_weight <= 0:
            return policy_loss, loss_dict

        target_prob = batch[self.type_target_key].to(z_type.device, dtype=z_type.dtype)
        if target_prob.ndim == 1:
            target_prob = target_prob.unsqueeze(0)
        logits = self.type_head(z_type)
        log_prob = F.log_softmax(logits, dim=-1)
        type_loss = -(target_prob * log_prob).sum(dim=-1).mean()
        lambda_type = self._current_aux_weight()
        total_loss = policy_loss + lambda_type * type_loss
        loss_dict = dict(loss_dict)
        loss_dict["loss_policy"] = float(policy_loss.detach().cpu())
        loss_dict["loss_type"] = float(type_loss.detach().cpu())
        loss_dict["lambda_type"] = float(lambda_type)
        loss_dict["type_acc"] = float((logits.argmax(dim=-1) == target_prob.argmax(dim=-1)).float().mean().cpu())
        self._train_step += 1
        return total_loss, loss_dict


def register_pi05_implicit_policy() -> None:
    import lerobot.policies.factory as factory_module

    original_get_policy_class = factory_module.get_policy_class
    original_make_processors = factory_module._make_processors_from_policy_config

    def patched_get_policy_class(name: str):
        if name == "pi05_implicit":
            return PI05ImplicitPolicy
        return original_get_policy_class(name)

    def patched_make_processors(config: PreTrainedConfig, dataset_stats=None):
        if isinstance(config, PI05ImplicitConfig):
            return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
        return original_make_processors(config=config, dataset_stats=dataset_stats)

    factory_module.get_policy_class = patched_get_policy_class
    factory_module._make_processors_from_policy_config = patched_make_processors
