from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import os

import packaging
import safetensors
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import Parameter

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
    action_mod_target: str = "actions"
    action_mod_mode: str = "gate_bias"
    action_mod_use_z_type: bool = True
    action_mod_use_z_task: bool = True
    action_mod_residual: bool = True
    state_concat_enabled: bool = True
    type_as_instruction: bool = False

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

        self.state_concat_enabled = bool(self.state_concat_enabled)
        self.type_as_instruction = bool(self.type_as_instruction)
        if self.action_mod_target not in {"action", "actions", "__disabled_train_action_mod__"}:
            raise ValueError("action_mod_target must be 'action', 'actions', or '__disabled_train_action_mod__'")
        if self.action_mod_mode != "gate_bias":
            raise ValueError("Only action_mod_mode='gate_bias' is supported")

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
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *args,
        config=None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | os.PathLike | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model = cls(config, **kwargs)
        model_id = str(pretrained_name_or_path)

        print(f"Loading model from: {pretrained_name_or_path}")
        if os.path.isdir(model_id):
            state_path = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            aux_path = os.path.join(model_id, 'implicit_aux_state.pt')
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"{SAFETENSORS_SINGLE_FILE} not found: {state_path}")
        else:
            try:
                state_path = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

            aux_path = None
            try:
                aux_path = hf_hub_download(
                    repo_id=model_id,
                    filename='implicit_aux_state.pt',
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError:
                pass

        original_state_dict = load_file(state_path)
        print(f"✓ Loaded state dict from {state_path}")

        fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)
        implicit_prefixes = (
            "implicit_conditioner.",
            "type_head.",
            "task_gate_head.",
            "task_bias_head.",
            "action_gate_head.",
            "action_bias_head.",
            "action_scale",
        )
        remapped_state_dict = {}
        remap_count = 0
        for key, value in fixed_state_dict.items():
            if key.startswith("model.") or key.startswith(implicit_prefixes):
                remapped_state_dict[key] = value
            else:
                new_key = f"model.{key}"
                remapped_state_dict[new_key] = value
                remap_count += 1
                if remap_count <= 10:
                    print(f"Remapped: {key} -> {new_key}")

        if remap_count > 0:
            print(f"Remapped {remap_count} state dict keys")

        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)
        if missing_keys:
            print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
            for key in missing_keys[:5]:
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        if unexpected_keys:
            print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
            for key in unexpected_keys[:5]:
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        if not missing_keys and not unexpected_keys:
            print("All keys loaded successfully!")

        if aux_path and os.path.exists(aux_path):
            state = torch.load(aux_path, map_location='cpu')
            model._train_step = int(state.get('train_step', 0))

        missing_implicit_keys = [
            key for key in missing_keys if key.startswith(("implicit_conditioner.", "type_head.", "task_gate_head.", "task_bias_head."))
        ]
        unexpected_implicit_keys = [
            key for key in unexpected_keys if key.startswith(("implicit_conditioner.", "type_head.", "task_gate_head.", "task_bias_head."))
        ]
        if missing_implicit_keys:
            print(f"Implicit-only keys missing from checkpoint (expected for base pi05 init): {len(missing_implicit_keys)} keys")
        if unexpected_implicit_keys:
            print(f"Unexpected implicit-only keys in checkpoint: {len(unexpected_implicit_keys)} keys")

        non_implicit_missing = [key for key in missing_keys if key not in missing_implicit_keys]
        non_implicit_unexpected = [key for key in unexpected_keys if key not in unexpected_implicit_keys]
        if strict and (non_implicit_missing or non_implicit_unexpected):
            raise RuntimeError(
                f"Non-implicit state dict mismatch. Missing: {len(non_implicit_missing)}, unexpected: {len(non_implicit_unexpected)}"
            )

        if not strict and missing_implicit_keys and len(missing_keys) == len(missing_implicit_keys) and not unexpected_keys:
            print("Proceeding with base pi05 weights; implicit-specific layers will stay randomly initialized.")

        if not strict and (non_implicit_missing or non_implicit_unexpected):
            print(f"Non-implicit mismatches remain. Missing: {len(non_implicit_missing)}, unexpected: {len(non_implicit_unexpected)}")
            for key in non_implicit_missing[:5]:
                print(f"  missing: {key}")
            for key in non_implicit_unexpected[:5]:
                print(f"  unexpected: {key}")

        model.to(config.device)
        model.eval()
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
        self.weak_coupling = config.weak_coupling
        self.weak_coupling_apply_at_inference = config.weak_coupling_apply_at_inference
        self.weak_coupling_gate_bias_init = config.weak_coupling_gate_bias_init
        self.weak_coupling_residual = config.weak_coupling_residual
        self.action_mod_target = config.action_mod_target
        self.action_mod_mode = config.action_mod_mode
        self.action_mod_use_z_type = config.action_mod_use_z_type
        self.action_mod_use_z_task = config.action_mod_use_z_task
        self.action_mod_residual = config.action_mod_residual
        self.state_concat_enabled = config.state_concat_enabled
        self.type_as_instruction = config.type_as_instruction
        self._train_step = 0
        super().__init__(config, **kwargs)
        self.implicit_conditioner = ImplicitConditioner(self.implicit_conditioning)
        self.implicit_conditioner.freeze_anchor_encoder()
        latent_dim = self.implicit_conditioner.latent_dim
        type_head_dim = self.z_type_dim if self.split_latent else latent_dim
        mod_dim = 0
        if self.action_mod_use_z_type:
            mod_dim += self.z_type_dim if self.split_latent else latent_dim
        if self.action_mod_use_z_task:
            mod_dim += self.z_task_dim if self.split_latent else latent_dim
        if mod_dim <= 0:
            mod_dim = latent_dim
        action_feature = self.config.output_features["action"]
        action_dim = int(action_feature.shape[0])
        self.type_head = nn.Linear(type_head_dim, config.type_num_classes)
        self.task_gate_head = nn.Linear(self.z_type_dim, self.z_task_dim)
        self.task_bias_head = nn.Linear(self.z_type_dim, self.z_task_dim)
        self.action_gate_head = nn.Linear(mod_dim, action_dim)
        self.action_bias_head = nn.Linear(mod_dim, action_dim)
        self.action_scale = Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.task_gate_head.weight)
        nn.init.constant_(self.task_gate_head.bias, self.weak_coupling_gate_bias_init)
        nn.init.zeros_(self.task_bias_head.weight)
        nn.init.zeros_(self.task_bias_head.bias)
        nn.init.zeros_(self.action_gate_head.weight)
        nn.init.zeros_(self.action_gate_head.bias)
        nn.init.zeros_(self.action_bias_head.weight)
        nn.init.zeros_(self.action_bias_head.bias)

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

    def _build_action_mod_signal(self, z_task: Tensor, z_type: Tensor) -> Tensor:
        parts: list[Tensor] = []
        if self.action_mod_use_z_type:
            parts.append(z_type)
        if self.action_mod_use_z_task:
            parts.append(z_task)
        if not parts:
            parts.append(z_type)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def _modulate_action_tensor(self, tensor: Tensor, mod_signal: Tensor) -> Tensor:
        gate = torch.sigmoid(self.action_gate_head(mod_signal)).unsqueeze(1)
        bias = self.action_bias_head(mod_signal).unsqueeze(1)
        scaled = self.action_scale.to(dtype=tensor.dtype, device=tensor.device)
        modulated = tensor * (1.0 + scaled * (gate - 0.5) * 2.0) + scaled * bias
        return tensor + modulated if self.action_mod_residual else modulated

    def _apply_implicit_conditioning(
        self,
        batch: dict[str, Tensor],
        *,
        apply_weak_coupling: bool,
        apply_action_modulation: bool,
    ) -> tuple[dict[str, Tensor], Tensor | None, Tensor | None, Tensor | None]:
        if not self.implicit_conditioning.enabled:
            return batch, None, None, None
        source_key = self.implicit_conditioner.source_image_key
        target_key = self.implicit_conditioner.target_state_key
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
        if self.state_concat_enabled:
            conditioned[target_key] = torch.cat([batch[target_key], z_task], dim=-1)
        if apply_action_modulation and self.action_mod_target in conditioned:
            mod_signal = self._build_action_mod_signal(z_task, z_type)
            conditioned[self.action_mod_target] = self._modulate_action_tensor(conditioned[self.action_mod_target], mod_signal)
        else:
            mod_signal = self._build_action_mod_signal(z_task, z_type)
        conditioned["_implicit_z_type"] = z_type
        conditioned["_implicit_z_task"] = z_task
        conditioned["_implicit_instruction"] = z_type if self.type_as_instruction else mod_signal
        return conditioned, z_type, z_task, mod_signal

    def _type_metrics(self, z_type: Tensor, batch: dict[str, Tensor], policy_loss: Tensor, loss_dict: dict) -> tuple[Tensor, dict]:
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
        return total_loss, loss_dict

    def _forward_with_action_modulation(self, conditioned: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict]:
        return super().forward(conditioned, reduction=reduction)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        conditioned, _, _, _ = self._apply_implicit_conditioning(
            batch,
            apply_weak_coupling=self.weak_coupling_apply_at_inference,
            apply_action_modulation=False,
        )
        return super().select_action(conditioned, **kwargs)

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        conditioned, _, _, _ = self._apply_implicit_conditioning(
            batch,
            apply_weak_coupling=self.weak_coupling_apply_at_inference,
            apply_action_modulation=False,
        )
        pred = super().predict_action_chunk(conditioned, **kwargs)
        z_task = conditioned.get("_implicit_z_task")
        z_type = conditioned.get("_implicit_z_type")
        if z_task is None or z_type is None:
            return pred
        mod_signal = self._build_action_mod_signal(z_task, z_type)
        return self._modulate_action_tensor(pred, mod_signal)

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        conditioned, z_type, _, _ = self._apply_implicit_conditioning(
            batch,
            apply_weak_coupling=True,
            apply_action_modulation=True,
        )
        policy_loss, loss_dict = self._forward_with_action_modulation(conditioned, reduction=reduction)
        total_loss, loss_dict = self._type_metrics(z_type, batch, policy_loss, loss_dict)
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
