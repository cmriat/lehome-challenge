from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import torch
from torch import nn


def append_condition_to_state(state: np.ndarray, condition: np.ndarray) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32)
    cond_arr = np.asarray(condition, dtype=np.float32)
    return np.concatenate([state_arr, cond_arr], axis=-1)


DEFAULT_LATENT_DIM = 8
ImplicitVariant = Literal["v1", "v2", "v3"]


@dataclass
class ImplicitConditioningConfig:
    enabled: bool = False
    variant: ImplicitVariant = "v1"
    latent_dim: int = DEFAULT_LATENT_DIM
    source_image_key: str = "observation.images.top_rgb"
    target_state_key: str = "observation.state"


class GarmentLatentEncoder(nn.Module):
    """Minimal visual encoder that produces an implicit garment latent from top RGB observations."""

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        top_key = "observation.images.top_rgb"
        if top_key not in images:
            raise ValueError(f"{top_key} is required for garment latent encoding")
        value = images[top_key]
        if value.ndim == 3:
            value = value.unsqueeze(0)
        x = value.float()
        h = self.backbone(x).flatten(1)
        return self.proj(h)


class ImplicitConditioner(nn.Module):
    def __init__(self, config: ImplicitConditioningConfig):
        super().__init__()
        self.config = config
        self.encoder = GarmentLatentEncoder(latent_dim=config.latent_dim)
        if config.variant == "v2":
            self.fusion_head = nn.Linear(config.latent_dim, config.latent_dim)
        elif config.variant == "v3":
            self.fusion_gate = nn.Sequential(nn.Linear(config.latent_dim, config.latent_dim), nn.Sigmoid())
            self.fusion_bias = nn.Linear(config.latent_dim, config.latent_dim)

    def encode(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent = self.encoder(images)
        if self.config.variant == "v2":
            latent = latent + self.fusion_head(latent)
        elif self.config.variant == "v3":
            latent = latent * self.fusion_gate(latent) + self.fusion_bias(latent)
        return latent

    def condition_state_tensor(self, state: torch.Tensor, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent = self.encode(images).to(dtype=state.dtype, device=state.device)
        return torch.cat([state, latent], dim=-1)


class ImplicitLeRobotPolicyMixin:
    implicit_conditioning: ImplicitConditioningConfig

    def _extract_image_tensors(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        top_key = self.implicit_conditioning.source_image_key
        if top_key not in observation:
            raise KeyError(f"{top_key} is required")
        value = observation[top_key]
        tensor = torch.from_numpy(value).to(self.device)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).float() / 255.0
        else:
            tensor = tensor.float()
        return {top_key: tensor.unsqueeze(0)}

    def _extract_latent_tensor(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        image_tensors = self._extract_image_tensors(observation)
        with torch.inference_mode():
            return self.implicit_conditioner.encode(image_tensors)

    def _build_conditioned_state(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        latent = self._extract_latent_tensor(observation)
        garment_latent = latent.squeeze(0).cpu().numpy().astype(np.float32)
        return append_condition_to_state(observation[self.implicit_conditioning.target_state_key], garment_latent)


def load_implicit_conditioning_config(policy_cfg) -> ImplicitConditioningConfig:
    raw = getattr(policy_cfg, "implicit_conditioning", None) or {}
    if not raw:
        return ImplicitConditioningConfig(enabled=False)
    if isinstance(raw, ImplicitConditioningConfig):
        return raw
    return ImplicitConditioningConfig(**raw)
