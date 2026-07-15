"""CNN-Transformer backbone and adversarial heads described by REALM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .config import ModelConfig


def sinusoidal_position_encoding(tokens: Tensor) -> Tensor:
    """Return a deterministic positional encoding matching ``[batch, time, width]``."""

    if tokens.ndim != 3:
        raise ValueError("tokens must be shaped [batch, time, width]")
    length, width = tokens.shape[1], tokens.shape[2]
    position = torch.arange(length, device=tokens.device, dtype=tokens.dtype).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=tokens.device, dtype=tokens.dtype)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros((length, width), device=tokens.device, dtype=tokens.dtype)
    encoding[:, 0::2] = torch.sin(position * frequency)
    if width > 1:
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, values: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = float(coefficient)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(values: Tensor, coefficient: float = 1.0) -> Tensor:
    """Identity in the forward pass and -coefficient times gradient backward."""

    return _GradientReversal.apply(values, coefficient)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class AttentionEncoderBlock(nn.Module):
    """Transformer encoder block that exposes per-head attention weights."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, return_attention: bool) -> tuple[Tensor, Tensor | None]:
        normalized = self.norm1(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        values = values + attended
        values = values + self.feedforward(self.norm2(values))
        return values, weights if return_attention else None


class CNNTransformerBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_channels = config.channels
        width = config.conv_width
        self.convolutions = nn.Sequential(
            ConvBlock(config.channels, width // 2, kernel_size=7, dropout=config.dropout),
            ConvBlock(width // 2, width, kernel_size=5, dropout=config.dropout),
            ConvBlock(width, width, kernel_size=3, dropout=config.dropout),
        )
        self.transformer = nn.ModuleList(
            AttentionEncoderBlock(
                width,
                config.attention_heads,
                config.feedforward_dim,
                config.dropout,
            )
            for _ in range(config.transformer_layers)
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.latent_dim),
            nn.GELU(),
        )

    def forward(
        self, values: Tensor, return_attention: bool = False
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if values.ndim != 3:
            raise ValueError("expected EEG tensor shaped [batch, channels, samples]")
        if values.shape[1] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} channels, received {values.shape[1]}")
        if values.shape[2] < 8:
            raise ValueError(
                "at least eight temporal samples are required for three pooling blocks"
            )
        tokens = self.convolutions(values).transpose(1, 2)
        tokens = tokens + sinusoidal_position_encoding(tokens)
        attention_maps: list[Tensor] = []
        for block in self.transformer:
            tokens, attention = block(tokens, return_attention)
            if attention is not None:
                attention_maps.append(attention)
        latent = self.projection(tokens.mean(dim=1))
        return latent, tuple(attention_maps)


@dataclass
class RealmOutput:
    seizure_logits: Tensor
    patient_logits: Tensor | None
    latent: Tensor
    attention: tuple[Tensor, ...]


class RealmModel(nn.Module):
    """Shared backbone with seizure and patient-identity classification heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = CNNTransformerBackbone(config)
        self.seizure_head = nn.Linear(config.latent_dim, 1)
        self.patient_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.patient_classes),
        )

    def forward(
        self,
        values: Tensor,
        grl_coefficient: float | None = None,
        return_attention: bool = False,
    ) -> RealmOutput:
        latent, attention = self.backbone(values, return_attention=return_attention)
        seizure_logits = self.seizure_head(latent).squeeze(-1)
        patient_logits = None
        if grl_coefficient is not None:
            patient_logits = self.patient_head(gradient_reverse(latent, grl_coefficient))
        return RealmOutput(seizure_logits, patient_logits, latent, attention)

    def predict_proba(self, values: Tensor) -> Tensor:
        """Return deterministic inference probabilities and restore module mode."""

        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                return torch.sigmoid(self(values).seizure_logits)
        finally:
            self.train(was_training)
