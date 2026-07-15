"""Stage 4: attribution, attention rollout, and latent-space positioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import RealmModel


@dataclass(frozen=True)
class ClassCentroids:
    non_seizure: Tensor
    seizure: Tensor


@dataclass
class ExplanationBundle:
    seizure_probability: Tensor
    input_attribution: Tensor
    temporal_attention: Tensor
    distance_to_non_seizure: Tensor | None
    distance_to_seizure: Tensor | None


def integrated_gradients(
    model: RealmModel,
    values: Tensor,
    baseline: Tensor | None = None,
    steps: int = 32,
) -> Tensor:
    """Approximate input attribution without requiring an external SHAP package."""

    if steps < 2:
        raise ValueError("steps must be at least 2")
    device = next(model.parameters()).device
    values = values.to(device=device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    try:
        if baseline is None:
            baseline = torch.zeros_like(values)
        else:
            baseline = baseline.to(device=device, dtype=values.dtype)
        if baseline.shape != values.shape:
            raise ValueError("baseline must have the same shape as values")
        gradients: list[Tensor] = []
        for alpha in torch.linspace(0.0, 1.0, steps, device=values.device):
            interpolated = (baseline + alpha * (values - baseline)).detach().requires_grad_(True)
            score = model(interpolated).seizure_logits.sum()
            gradient = torch.autograd.grad(score, interpolated)[0]
            gradients.append(gradient)
        average_gradient = torch.stack(gradients).mean(dim=0)
        return (values - baseline) * average_gradient
    finally:
        model.train(was_training)


def shap_attribution(
    model: RealmModel,
    values: Tensor,
    background: Tensor,
) -> Tensor:
    """Compute GradientExplainer SHAP values when the optional package is installed."""

    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP attribution requires realm-eeg[explain]") from exc

    class _SeizureLogit(torch.nn.Module):
        def __init__(self, wrapped: RealmModel) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: Tensor) -> Tensor:
            return self.wrapped(x).seizure_logits.unsqueeze(-1)

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        values = values.to(device=device, dtype=torch.float32)
        background = background.to(device=device, dtype=torch.float32)
        explainer = shap.GradientExplainer(_SeizureLogit(model), background)
        shap_values = explainer.shap_values(values)
        values_array = shap_values
        if isinstance(shap_values, list):
            values_array = shap_values[0]
        return torch.as_tensor(values_array, device=values.device, dtype=values.dtype).squeeze(-1)
    finally:
        model.train(was_training)


def attention_rollout(attention_maps: tuple[Tensor, ...]) -> Tensor:
    """Aggregate residual-aware self-attention into temporal token importance."""

    if not attention_maps:
        raise ValueError("the model must be called with return_attention=True")
    batch, _, tokens, _ = attention_maps[0].shape
    identity = torch.eye(tokens, device=attention_maps[0].device).expand(batch, -1, -1)
    rollout = identity
    for layer in attention_maps:
        mean_heads = layer.mean(dim=1)
        residual = mean_heads + identity
        residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rollout = residual @ rollout
    return rollout.mean(dim=1)


def fit_centroids(embeddings: Tensor, labels: Tensor) -> ClassCentroids:
    labels = labels.to(device=embeddings.device, dtype=torch.long)
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("both seizure and non-seizure embeddings are required")
    return ClassCentroids(
        non_seizure=embeddings[labels == 0].mean(dim=0),
        seizure=embeddings[labels == 1].mean(dim=0),
    )


def centroid_distances(embeddings: Tensor, centroids: ClassCentroids) -> tuple[Tensor, Tensor]:
    non_seizure_centroid = centroids.non_seizure.to(embeddings)
    seizure_centroid = centroids.seizure.to(embeddings)
    non_seizure = torch.linalg.vector_norm(embeddings - non_seizure_centroid, dim=-1)
    seizure = torch.linalg.vector_norm(embeddings - seizure_centroid, dim=-1)
    return non_seizure, seizure


def explain_prediction(
    model: RealmModel,
    values: Tensor,
    centroids: ClassCentroids | None = None,
    integration_steps: int = 32,
) -> ExplanationBundle:
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        values = values.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model(values, return_attention=True)
        attribution = integrated_gradients(model, values, steps=integration_steps)
        temporal_attention = attention_rollout(output.attention)
        non_seizure_distance = seizure_distance = None
        if centroids is not None:
            non_seizure_distance, seizure_distance = centroid_distances(output.latent, centroids)
        return ExplanationBundle(
            seizure_probability=torch.sigmoid(output.seizure_logits),
            input_attribution=attribution,
            temporal_attention=temporal_attention,
            distance_to_non_seizure=non_seizure_distance,
            distance_to_seizure=seizure_distance,
        )
    finally:
        model.train(was_training)
