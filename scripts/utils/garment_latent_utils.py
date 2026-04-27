from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn


def append_condition_to_state(state: np.ndarray, condition: np.ndarray) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32)
    cond_arr = np.asarray(condition, dtype=np.float32)
    return np.concatenate([state_arr, cond_arr], axis=-1)


DEFAULT_LATENT_DIM = 8
ImplicitVariant = Literal["v1", "v2", "v3", "v4", "v5"]


@dataclass
class ImplicitConditioningConfig:
    enabled: bool = False
    variant: ImplicitVariant = "v1"
    latent_dim: int = DEFAULT_LATENT_DIM
    source_image_key: str = "observation.images.top_rgb"
    target_state_key: str = "observation.state"
    anchor_encoder_name_or_path: str = "vit_base_patch14_dinov2.lvd142m"
    anchor_pooling: Literal["cls", "mean", "attn"] = "attn"
    anchor_attn_num_queries: int = 4
    anchor_attn_num_heads: int = 8
    anchor_feat_dim: int = 768
    anchor_proj_dim: int = DEFAULT_LATENT_DIM
    anchor_proj_hidden_dims: list = field(default_factory=lambda: [256, 64])
    freeze_anchor_encoder: bool = True
    lora_enabled: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_blocks: int = 4
    type_as_instruction: bool = False
    state_concat_enabled: bool = True


class AttentionPool(nn.Module):
    """Learnable cross-attention pooling over patch tokens.

    Replaces mean pooling with a set of learnable query vectors that attend
    to patch features, letting the model learn which spatial regions matter.
    """

    def __init__(self, dim: int, num_queries: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = patch_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, patch_tokens, patch_tokens)
        return out.flatten(1)


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank if rank > 0 else 1.0
        in_features, out_features = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if rank <= 0:
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.lora_A) @ self.lora_B * self.scale
        return self.linear(x) + delta


def _apply_lora_to_blocks(encoder, num_blocks: int, rank: int, alpha: float):
    """Inject LoRA into the last `num_blocks` transformer blocks of a ViT."""
    if rank <= 0 or num_blocks <= 0:
        return
    blocks = encoder.blocks
    target_blocks = blocks[-num_blocks:] if num_blocks < len(blocks) else blocks
    for block in target_blocks:
        attn = block.attn
        attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)


class GarmentLatentEncoder(nn.Module):
    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

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


class PretrainedGarmentEncoder(nn.Module):
    """Pretrained visual encoder with learnable attention pooling and MLP projection.

    Uses a frozen (or LoRA-tuned) pretrained ViT backbone, then pools patch tokens
    via cross-attention with learnable queries, and projects via MLP to the latent space.
    """

    def __init__(self, config: ImplicitConditioningConfig):
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            config.anchor_encoder_name_or_path,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        if config.freeze_anchor_encoder:
            self.encoder.requires_grad_(False)
        if config.lora_enabled:
            _apply_lora_to_blocks(
                self.encoder, config.lora_target_blocks, config.lora_rank, config.lora_alpha
            )

        if config.anchor_pooling == "attn":
            self.attn_pool = AttentionPool(
                dim=config.anchor_feat_dim,
                num_queries=config.anchor_attn_num_queries,
                num_heads=config.anchor_attn_num_heads,
            )
            pooled_dim = config.anchor_attn_num_queries * config.anchor_feat_dim
        else:
            self.attn_pool = None
            pooled_dim = config.anchor_feat_dim

        hidden = config.anchor_proj_hidden_dims
        dims = [pooled_dim] + list(hidden) + [config.anchor_proj_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        if features.ndim == 2:
            return features
        if features.ndim != 3:
            raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")

        if self.attn_pool is not None:
            patch_tokens = features[:, 1:] if features.shape[1] > self.encoder.num_prefix_tokens else features
            return self.attn_pool(patch_tokens)

        if self.config.anchor_pooling == "cls":
            return features[:, 0]
        return features.mean(dim=1)

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        top_key = self.config.source_image_key
        if top_key not in images:
            raise ValueError(f"{top_key} is required for garment latent encoding")
        value = images[top_key]
        if value.ndim == 3:
            value = value.unsqueeze(0)
        x = value.float()
        expected_size = getattr(self.encoder.patch_embed, "img_size", None)
        if expected_size is not None:
            if isinstance(expected_size, tuple):
                target_h, target_w = expected_size
            else:
                target_h = target_w = int(expected_size)
            if x.shape[-2] != target_h or x.shape[-1] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        features = self.encoder.forward_features(x)
        pooled = self._pool_features(features)
        return self.proj(pooled)


class ImplicitConditioner(nn.Module):
    def __init__(self, config: ImplicitConditioningConfig):
        super().__init__()
        self.config = config
        self.encoder = (
            PretrainedGarmentEncoder(config)
            if config.variant == "v5"
            else GarmentLatentEncoder(latent_dim=config.latent_dim)
        )
        if config.variant == "v2":
            self.fusion_head = nn.Linear(config.latent_dim, config.latent_dim)
        elif config.variant in {"v3", "v4"}:
            self.fusion_gate = nn.Sequential(nn.Linear(config.latent_dim, config.latent_dim), nn.Sigmoid())
            self.fusion_bias = nn.Linear(config.latent_dim, config.latent_dim)

    @property
    def latent_dim(self) -> int:
        return self.config.anchor_proj_dim if self.config.variant == "v5" else self.config.latent_dim

    @property
    def source_image_key(self) -> str:
        return self.config.source_image_key

    @property
    def target_state_key(self) -> str:
        return self.config.target_state_key

    def freeze_anchor_encoder(self) -> None:
        if hasattr(self.encoder, "encoder") and self.config.freeze_anchor_encoder:
            self.encoder.encoder.requires_grad_(False)

    def encode(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent = self.encoder(images)
        if self.config.variant == "v2":
            latent = latent + self.fusion_head(latent)
        elif self.config.variant in {"v3", "v4"}:
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


def resolve_implicit_variant(policy_cfg: Any) -> str | None:
    raw = getattr(policy_cfg, "implicit_conditioning", None) or {}
    if not raw:
        return None
    return raw.get("variant") if isinstance(raw, dict) else getattr(raw, "variant", None)


def load_implicit_conditioning_config(policy_cfg) -> ImplicitConditioningConfig:
    raw = getattr(policy_cfg, "implicit_conditioning", None) or {}
    if not raw:
        return ImplicitConditioningConfig(enabled=False)
    if isinstance(raw, ImplicitConditioningConfig):
        return raw
    return ImplicitConditioningConfig(**raw)
