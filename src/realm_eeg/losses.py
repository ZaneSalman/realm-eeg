"""Losses and schedules for the first two REALM stages."""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor, nn

from .model import RealmOutput


@dataclass
class LossBreakdown:
    total: Tensor
    seizure: Tensor
    patient: Tensor | None = None


def dann_coefficient(progress: float, steepness: float = 10.0) -> float:
    """Smoothly increase the GRL coefficient from zero toward one."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be between 0 and 1")
    if steepness <= 0:
        raise ValueError("steepness must be positive")
    return 2.0 / (1.0 + math.exp(-steepness * progress)) - 1.0


def stage1_loss(output: RealmOutput, seizure_labels: Tensor) -> LossBreakdown:
    labels = seizure_labels.to(dtype=output.seizure_logits.dtype)
    seizure = nn.functional.binary_cross_entropy_with_logits(output.seizure_logits, labels)
    return LossBreakdown(total=seizure, seizure=seizure)


def stage2_loss(
    output: RealmOutput,
    seizure_labels: Tensor,
    patient_labels: Tensor,
    patient_weight: float = 1.0,
) -> LossBreakdown:
    if patient_weight < 0:
        raise ValueError("patient_weight cannot be negative")
    if output.patient_logits is None:
        raise ValueError("Stage 2 requires a non-null grl_coefficient during forward")
    labels = seizure_labels.to(dtype=output.seizure_logits.dtype)
    seizure = nn.functional.binary_cross_entropy_with_logits(output.seizure_logits, labels)
    patient = nn.functional.cross_entropy(output.patient_logits, patient_labels.long())
    # The GRL already reverses and scales the patient gradient seen by the backbone.
    # Adding both losses trains the patient head normally while making the backbone adversarial.
    total = seizure + patient_weight * patient
    return LossBreakdown(total=total, seizure=seizure, patient=patient)
