"""Deployment-time few-shot adaptation for a previously unseen patient."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .model import RealmModel


def adapt_to_patient(
    model: RealmModel,
    support_x: Tensor,
    support_y: Tensor,
    steps: int = 5,
    learning_rate: float = 1e-3,
    adapt_backbone: bool = True,
) -> RealmModel:
    """Return an adapted copy; the shared source model is never mutated."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if support_x.shape[0] != support_y.shape[0]:
        raise ValueError("support_x and support_y must contain the same number of examples")
    if support_y.numel() == 0:
        raise ValueError("support set cannot be empty")
    if not torch.all((support_y == 0) | (support_y == 1)):
        raise ValueError("support_y must contain only binary labels")
    adapted = copy.deepcopy(model)
    device = next(adapted.parameters()).device
    support_x = support_x.to(device=device, dtype=torch.float32)
    support_y = support_y.to(device=device, dtype=torch.float32)
    adapted.train()
    for parameter in adapted.patient_head.parameters():
        parameter.requires_grad_(False)
    if not adapt_backbone:
        for parameter in adapted.backbone.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = adapted(support_x).seizure_logits
        loss = nn.functional.binary_cross_entropy_with_logits(logits, support_y)
        loss.backward()
        optimizer.step()
    adapted.eval()
    return adapted
