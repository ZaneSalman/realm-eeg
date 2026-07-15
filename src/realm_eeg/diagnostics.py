"""Post-training diagnostics for patient information in frozen embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PatientProbeResult:
    """Held-out accuracy and transparent no-skill baselines for a linear probe."""

    accuracy: float
    majority_baseline_accuracy: float
    uniform_random_baseline_accuracy: float
    majority_patient_id: int
    patient_classes: tuple[int, ...]
    train_examples: int
    test_examples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _as_tensor(name: str, values: Tensor | np.ndarray) -> Tensor:
    try:
        return torch.as_tensor(values)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


def _integer_vector(name: str, values: Tensor | np.ndarray) -> Tensor:
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if values.size and np.issubdtype(values.dtype, np.unsignedinteger):
            if values.max() > np.iinfo(np.int64).max:
                raise ValueError(f"{name} contains an integer outside the supported range")
        return torch.from_numpy(np.array(values, dtype=np.int64, copy=True))
    vector = _as_tensor(name, values)
    if vector.ndim != 1 or vector.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return vector.detach().to(device="cpu", dtype=torch.long).clone()


def _probe_indices(
    name: str,
    values: Tensor | np.ndarray,
    *,
    observations: int,
) -> Tensor:
    indices = _integer_vector(name, values)
    if indices.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if (indices < 0).any() or (indices >= observations).any():
        raise ValueError(f"{name} contains an out-of-range index")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"{name} cannot contain duplicate indices")
    return indices


def patient_identity_linear_probe(
    embeddings: Tensor | np.ndarray,
    patient_ids: Tensor | np.ndarray,
    *,
    train_indices: Tensor | np.ndarray,
    test_indices: Tensor | np.ndarray,
    steps: int = 400,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
) -> PatientProbeResult:
    """Fit and evaluate a deterministic linear patient-identity diagnostic.

    The caller supplies precomputed embeddings and an explicit held-out split.
    Embeddings are detached, copied, and fitted on CPU in double precision; the
    representation model is never updated. The full training split is used for
    each deterministic optimization step (there is no minibatch randomness).

    ``majority_baseline_accuracy`` evaluates an always-predict-the-training-
    majority classifier on the test split. Ties are resolved by the smallest
    patient ID. ``uniform_random_baseline_accuracy`` is the expected accuracy of
    choosing uniformly among patient classes, namely ``1 / number_of_classes``.

    A near-baseline result only means that this particular linear probe did not
    recover identity. It is not proof that the representation is invariant or
    that a nonlinear attacker could not recover patient information.
    """

    features = _as_tensor("embeddings", embeddings)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array [examples, features]")
    if features.dtype == torch.bool or features.is_complex():
        raise ValueError("embeddings must contain real numeric values")
    if not torch.isfinite(features).all():
        raise ValueError("embeddings must contain only finite values")
    features = features.detach().to(device="cpu", dtype=torch.float64).clone()

    identities = _integer_vector("patient_ids", patient_ids)
    if identities.shape != (features.shape[0],):
        raise ValueError("patient_ids must have one value per embedding")
    if (identities < 0).any():
        raise ValueError("patient_ids cannot contain negative values")

    train = _probe_indices("train_indices", train_indices, observations=features.shape[0])
    test = _probe_indices("test_indices", test_indices, observations=features.shape[0])
    if torch.isin(train, test).any():
        raise ValueError("train_indices and test_indices must be disjoint")

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")

    train_patient_ids = identities[train]
    test_patient_ids = identities[test]
    patient_classes = torch.unique(train_patient_ids, sorted=True)
    if patient_classes.numel() < 2:
        raise ValueError("the training split must contain at least two patient classes")
    missing_test_classes = torch.unique(
        test_patient_ids[~torch.isin(test_patient_ids, patient_classes)]
    )
    if missing_test_classes.numel():
        missing = [int(value) for value in missing_test_classes.tolist()]
        raise ValueError(f"test patient classes are absent from training: {missing}")

    train_targets = torch.searchsorted(patient_classes, train_patient_ids)
    test_targets = torch.searchsorted(patient_classes, test_patient_ids)
    train_features = features[train]
    test_features = features[test]

    # Module construction normally consumes the caller's global RNG even though
    # this probe uses zero initialization. Forking preserves that external state.
    with torch.random.fork_rng(devices=[]):
        probe = nn.Linear(features.shape[1], patient_classes.numel(), dtype=torch.float64)
        nn.init.zeros_(probe.weight)
        nn.init.zeros_(probe.bias)
        optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(probe(train_features), train_targets)
            loss.backward()
            optimizer.step()

    with torch.inference_mode():
        predictions = probe(test_features).argmax(dim=1)
        accuracy = float((predictions == test_targets).double().mean())

    class_counts = torch.bincount(train_targets, minlength=patient_classes.numel())
    majority_class_index = int(class_counts.argmax())
    majority_patient_id = int(patient_classes[majority_class_index])
    majority_baseline = float((test_patient_ids == majority_patient_id).double().mean())
    uniform_baseline = 1.0 / int(patient_classes.numel())

    return PatientProbeResult(
        accuracy=accuracy,
        majority_baseline_accuracy=majority_baseline,
        uniform_random_baseline_accuracy=uniform_baseline,
        majority_patient_id=majority_patient_id,
        patient_classes=tuple(int(value) for value in patient_classes.tolist()),
        train_examples=int(train.numel()),
        test_examples=int(test.numel()),
    )
