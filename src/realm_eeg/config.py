"""Typed, JSON-serializable configuration objects for REALM."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from numbers import Real
from typing import Any, Mapping


def _require_integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        requirement = "a positive integer" if minimum == 1 else f"an integer of at least {minimum}"
        raise ValueError(f"{name} must be {requirement}")


def _require_finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None:
        invalid = value <= minimum if strict_minimum else value < minimum
        if invalid:
            comparison = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {comparison} {minimum}")


def _validate_compatibility(data: "DataConfig", model: "ModelConfig") -> None:
    if data.channels != model.channels:
        raise ValueError(
            "data.channels must equal model.channels; harmonize the input contract "
            "or change both settings together"
        )


@dataclass(frozen=True)
class DataConfig:
    sampling_rate_hz: float = 256.0
    segment_seconds: float = 4.0
    channels: int = 19
    normalize_epsilon: float = 1e-6

    @property
    def samples_per_segment(self) -> int:
        return round(self.sampling_rate_hz * self.segment_seconds)

    def validate(self) -> None:
        _require_finite_real(
            "sampling_rate_hz", self.sampling_rate_hz, minimum=0, strict_minimum=True
        )
        _require_finite_real(
            "segment_seconds", self.segment_seconds, minimum=0, strict_minimum=True
        )
        _require_integer("channels", self.channels, 1)
        _require_finite_real(
            "normalize_epsilon", self.normalize_epsilon, minimum=0, strict_minimum=True
        )
        if self.samples_per_segment < 1:
            raise ValueError("sampling_rate_hz * segment_seconds must produce at least one sample")


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 19
    conv_width: int = 64
    latent_dim: int = 128
    transformer_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    patient_classes: int = 22

    def validate(self) -> None:
        _require_integer("channels", self.channels, 1)
        _require_integer("conv_width", self.conv_width, 2)
        _require_integer("latent_dim", self.latent_dim, 1)
        _require_integer("transformer_layers", self.transformer_layers, 1)
        _require_integer("attention_heads", self.attention_heads, 1)
        _require_integer("feedforward_dim", self.feedforward_dim, 1)
        _require_integer("patient_classes", self.patient_classes, 2)
        if self.conv_width % self.attention_heads:
            raise ValueError("conv_width must be divisible by attention_heads")
        _require_finite_real("dropout", self.dropout, minimum=0)
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    stage1_epochs: int = 20
    stage2_epochs: int = 20
    inner_learning_rate: float = 1e-2
    inner_steps: int = 1
    first_order_maml: bool = True
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        _require_finite_real("learning_rate", self.learning_rate, minimum=0, strict_minimum=True)
        _require_finite_real("weight_decay", self.weight_decay, minimum=0)
        _require_integer("stage1_epochs", self.stage1_epochs, 1)
        _require_integer("stage2_epochs", self.stage2_epochs, 1)
        _require_finite_real(
            "inner_learning_rate", self.inner_learning_rate, minimum=0, strict_minimum=True
        )
        _require_integer("inner_steps", self.inner_steps, 1)
        if not isinstance(self.first_order_maml, bool):
            raise ValueError("first_order_maml must be a Boolean")
        _require_finite_real(
            "gradient_clip_norm", self.gradient_clip_norm, minimum=0, strict_minimum=True
        )


def load_config(path: str | Path) -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = DataConfig(**payload.get("data", {}))
    model = ModelConfig(**payload.get("model", {}))
    training = TrainingConfig(**payload.get("training", {}))
    data.validate()
    model.validate()
    training.validate()
    _validate_compatibility(data, model)
    return data, model, training


def save_config(
    path: str | Path,
    data: DataConfig,
    model: ModelConfig,
    training: TrainingConfig,
    extra: Mapping[str, Any] | None = None,
) -> None:
    data.validate()
    model.validate()
    training.validate()
    _validate_compatibility(data, model)
    payload: dict[str, Any] = {
        "data": asdict(data),
        "model": asdict(model),
        "training": asdict(training),
    }
    if extra:
        payload["extra"] = dict(extra)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
