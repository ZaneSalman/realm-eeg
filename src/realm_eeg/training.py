"""Training loops for population pretraining, disentanglement, and meta-learning."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .config import ModelConfig
from .data import Episode
from .losses import dann_coefficient, stage1_loss, stage2_loss
from .model import RealmModel


def _unpack_batch(batch: Any, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(batch, Mapping):
        x, y, patient = batch["x"], batch["y"], batch["patient_id"]
    elif isinstance(batch, Sequence) and len(batch) == 3:
        x, y, patient = batch
    else:
        raise TypeError("batch must be a mapping or (x, y, patient_id) sequence")
    return (
        x.to(device=device, dtype=torch.float32),
        y.to(device=device, dtype=torch.float32),
        patient.to(device=device, dtype=torch.long),
    )


def train_stage1_epoch(
    model: RealmModel,
    batches: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    examples = 0
    for batch in batches:
        x, y, _ = _unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        breakdown = stage1_loss(model(x), y)
        breakdown.total.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += float(breakdown.total.detach()) * x.shape[0]
        examples += x.shape[0]
    if not examples:
        raise ValueError("Stage 1 received no examples")
    return {"loss": total_loss / examples}


def train_stage2_epoch(
    model: RealmModel,
    batches: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    total_epochs: int,
    device: torch.device | str = "cpu",
    patient_weight: float = 1.0,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Train with a positive patient loss and a GRL-scaled backbone gradient."""

    if total_epochs < 1 or not 0 <= epoch_index < total_epochs:
        raise ValueError("epoch_index must be within total_epochs")
    model.train()
    totals = {"loss": 0.0, "seizure_loss": 0.0, "patient_loss": 0.0}
    examples = 0
    try:
        batches_per_epoch = len(batches)  # type: ignore[arg-type]
    except TypeError:
        batches_per_epoch = None
    coefficients: list[float] = []
    for batch_index, batch in enumerate(batches):
        if batches_per_epoch is not None:
            total_steps = total_epochs * batches_per_epoch
            global_step = epoch_index * batches_per_epoch + batch_index
            progress = global_step / (total_steps - 1) if total_steps > 1 else 0.0
        else:
            progress = epoch_index / (total_epochs - 1) if total_epochs > 1 else 0.0
        coefficient = dann_coefficient(progress)
        coefficients.append(coefficient)
        x, y, patient = _unpack_batch(batch, device)
        if patient.numel() and (
            int(patient.min()) < 0 or int(patient.max()) >= model.config.patient_classes
        ):
            raise ValueError(
                "Stage 2 patient labels must be remapped to contiguous values in "
                f"[0, {model.config.patient_classes - 1}]"
            )
        optimizer.zero_grad(set_to_none=True)
        output = model(x, grl_coefficient=coefficient)
        breakdown = stage2_loss(output, y, patient, patient_weight=patient_weight)
        breakdown.total.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = x.shape[0]
        totals["loss"] += float(breakdown.total.detach()) * batch_size
        totals["seizure_loss"] += float(breakdown.seizure.detach()) * batch_size
        assert breakdown.patient is not None
        totals["patient_loss"] += float(breakdown.patient.detach()) * batch_size
        examples += batch_size
    if not examples:
        raise ValueError("Stage 2 received no examples")
    return {key: value / examples for key, value in totals.items()} | {
        "grl_coefficient": coefficients[-1],
        "grl_coefficient_start": coefficients[0],
    }


def _functional_forward(
    model: RealmModel,
    parameters: OrderedDict[str, Tensor],
    buffers: Mapping[str, Tensor],
    x: Tensor,
):
    state = dict(buffers)
    state.update(parameters)
    return functional_call(model, state, (x,))


def maml_episode_loss(
    model: RealmModel,
    episode: Episode,
    inner_learning_rate: float = 1e-2,
    inner_steps: int = 1,
    first_order: bool = True,
    adapt_backbone: bool = True,
) -> Tensor:
    """Compute the query loss after differentiable support-set adaptation.

    REALM does not specify whether the seizure head is included in theta. This
    reference adapts the seizure head and, by default, the shared backbone.
    """

    if inner_steps < 1:
        raise ValueError("inner_steps must be positive")
    if inner_learning_rate <= 0:
        raise ValueError("inner_learning_rate must be positive")
    if episode.support_x.shape[0] != episode.support_y.shape[0] or not episode.support_y.numel():
        raise ValueError("support inputs and labels must be non-empty and aligned")
    if episode.query_x.shape[0] != episode.query_y.shape[0] or not episode.query_y.numel():
        raise ValueError("query inputs and labels must be non-empty and aligned")
    device = next(model.parameters()).device
    support_x = episode.support_x.to(device=device, dtype=torch.float32)
    support_y = episode.support_y.to(device=device, dtype=torch.float32)
    query_x = episode.query_x.to(device=device, dtype=torch.float32)
    query_y = episode.query_y.to(device=device, dtype=torch.float32)
    parameters = OrderedDict(model.named_parameters())
    buffers = dict(model.named_buffers())
    adapt_names = [
        name
        for name in parameters
        if not name.startswith("patient_head.")
        and (adapt_backbone or name.startswith("seizure_head."))
    ]
    fast = parameters
    for _ in range(inner_steps):
        support_output = _functional_forward(model, fast, buffers, support_x)
        support_loss = nn.functional.binary_cross_entropy_with_logits(
            support_output.seizure_logits, support_y
        )
        gradients = torch.autograd.grad(
            support_loss,
            [fast[name] for name in adapt_names],
            create_graph=not first_order,
        )
        if first_order:
            gradients = tuple(gradient.detach() for gradient in gradients)
        updated = OrderedDict(fast)
        for name, gradient in zip(adapt_names, gradients):
            updated[name] = fast[name] - inner_learning_rate * gradient
        fast = updated
    query_output = _functional_forward(model, fast, buffers, query_x)
    return nn.functional.binary_cross_entropy_with_logits(query_output.seizure_logits, query_y)


def train_meta_epoch(
    model: RealmModel,
    episodes: Iterable[Episode],
    optimizer: torch.optim.Optimizer,
    inner_learning_rate: float = 1e-2,
    inner_steps: int = 1,
    first_order: bool = True,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    episode_list = list(episodes)
    if not episode_list:
        raise ValueError("meta-training received no valid patient episodes")
    optimizer.zero_grad(set_to_none=True)
    meta_loss_value = 0.0
    for episode in episode_list:
        loss = maml_episode_loss(
            model,
            episode,
            inner_learning_rate=inner_learning_rate,
            inner_steps=inner_steps,
            first_order=first_order,
        )
        meta_loss_value += float(loss.detach())
        (loss / len(episode_list)).backward()
    if gradient_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return {
        "meta_loss": meta_loss_value / len(episode_list),
        "episodes": float(len(episode_list)),
    }


@torch.no_grad()
def extract_embeddings(
    model: RealmModel,
    batches: Iterable[Any],
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    patients: list[Tensor] = []
    for batch in batches:
        x, y, patient = _unpack_batch(batch, device)
        embeddings.append(model(x).latent.cpu())
        labels.append(y.cpu())
        patients.append(patient.cpu())
    return torch.cat(embeddings), torch.cat(labels), torch.cat(patients)


def save_checkpoint(
    path: str | Path,
    model: RealmModel,
    stage: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "software_version": "0.1.0",
        "stage": stage,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
    }
    if extra:
        extra_payload = dict(extra)
        _validate_weights_only_value(extra_payload, "extra")
        payload["extra"] = extra_payload
    torch.save(payload, path)


def _validate_weights_only_value(value: Any, location: str) -> None:
    """Reject checkpoint metadata that the safe weights-only loader cannot read."""

    if value is None or isinstance(value, (bool, int, float, str, Tensor)):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_weights_only_value(item, f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} checkpoint metadata keys must be strings")
            _validate_weights_only_value(item, f"{location}.{key}")
        return
    raise TypeError(
        f"{location} contains unsupported checkpoint metadata type {type(value).__name__}"
    )


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[RealmModel, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported or malformed REALM checkpoint")
    if "model_config" not in payload or "state_dict" not in payload:
        raise ValueError("checkpoint is missing model_config or state_dict")
    model = RealmModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    return model, payload
