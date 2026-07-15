"""Window datasets, leak-safe patient splits, and synthetic smoke-test data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

_REQUIRED_ARRAYS = frozenset({"x", "y", "patient_id"})
_OPTIONAL_ALIGNED_ARRAYS = ("recording_id", "group_id", "window_start_seconds")


@dataclass(frozen=True)
class PatientSplit:
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    train_patients: tuple[int, ...]
    validation_patients: tuple[int, ...]
    test_patients: tuple[int, ...]


@dataclass(frozen=True)
class Episode:
    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor
    patient_id: int


class NPZWindowDataset(Dataset[Mapping[str, Tensor]]):
    """Read the documented NPZ interchange format into memory.

    Required arrays:
      x: float [windows, channels, samples]
      y: binary [windows]
      patient_id: integer [windows]

    Optional aligned arrays (all three must be present together):
      recording_id: non-negative integer [windows]
      group_id: non-negative integer [windows]
      window_start_seconds: non-negative finite float [windows]

    NumPy cannot memory-map members of a ZIP archive. Use
    :class:`NPYDirectoryDataset` for clinical-scale arrays.
    """

    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            validate_npz_arrays(payload)
            self.x = payload["x"]
            self.y = payload["y"]
            self.patient_id = payload["patient_id"]
            self.recording_id = payload.get("recording_id")
            self.group_id = payload.get("group_id")
            self.window_start_seconds = payload.get("window_start_seconds")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Mapping[str, Tensor]:
        item = {
            "x": torch.as_tensor(np.array(self.x[index], copy=True), dtype=torch.float32),
            "y": torch.as_tensor(self.y[index], dtype=torch.float32),
            "patient_id": torch.as_tensor(self.patient_id[index], dtype=torch.long),
        }
        if self.recording_id is not None:
            item.update(
                {
                    "recording_id": torch.as_tensor(self.recording_id[index], dtype=torch.long),
                    "group_id": torch.as_tensor(self.group_id[index], dtype=torch.long),
                    "window_start_seconds": torch.as_tensor(
                        self.window_start_seconds[index], dtype=torch.float64
                    ),
                }
            )
        return item


class NPYDirectoryDataset(NPZWindowDataset):
    """Memory-map required and optional aligned ``.npy`` arrays from a directory."""

    def __init__(self, path: str | Path) -> None:
        directory = Path(path)
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("x", "y", "patient_id")
        }
        optional_paths = {name: directory / f"{name}.npy" for name in _OPTIONAL_ALIGNED_ARRAYS}
        present_optional = {name for name, path in optional_paths.items() if path.is_file()}
        if present_optional and len(present_optional) != len(optional_paths):
            missing = sorted(set(optional_paths) - present_optional)
            raise ValueError(
                "optional aligned arrays must be provided together; "
                f"missing files: {[f'{name}.npy' for name in missing]}"
            )
        if present_optional:
            arrays.update(
                {
                    name: np.load(path, mmap_mode="r", allow_pickle=False)
                    for name, path in optional_paths.items()
                }
            )
        validate_npz_arrays(arrays)
        self.x = arrays["x"]
        self.y = arrays["y"]
        self.patient_id = arrays["patient_id"]
        self.recording_id = arrays.get("recording_id")
        self.group_id = arrays.get("group_id")
        self.window_start_seconds = arrays.get("window_start_seconds")


def validate_npz_arrays(payload: Mapping[str, NDArray[np.generic]]) -> None:
    missing = _REQUIRED_ARRAYS - set(payload)
    if missing:
        raise ValueError(f"NPZ is missing required arrays: {sorted(missing)}")
    present_optional = set(_OPTIONAL_ALIGNED_ARRAYS) & set(payload)
    if present_optional and len(present_optional) != len(_OPTIONAL_ALIGNED_ARRAYS):
        missing_optional = sorted(set(_OPTIONAL_ALIGNED_ARRAYS) - present_optional)
        raise ValueError(
            f"optional aligned arrays must be provided together; missing: {missing_optional}"
        )
    x = np.asarray(payload["x"])
    y = np.asarray(payload["y"])
    patient = np.asarray(payload["patient_id"])
    if x.ndim != 3:
        raise ValueError("x must be shaped [windows, channels, samples]")
    if 0 in x.shape:
        raise ValueError("x cannot contain an empty window, channel, or sample axis")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("x must use a real floating-point dtype")
    if y.shape != (x.shape[0],) or patient.shape != (x.shape[0],):
        raise ValueError("y and patient_id must have one value per window")
    for start in range(0, x.shape[0], 1024):
        if not np.isfinite(x[start : start + 1024]).all():
            raise ValueError("x contains NaN or infinite values")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1")
    if not np.issubdtype(patient.dtype, np.integer):
        raise ValueError("patient_id must be an integer array")
    if (patient < 0).any():
        raise ValueError("patient_id cannot contain negative values")
    if present_optional:
        recording = np.asarray(payload["recording_id"])
        group = np.asarray(payload["group_id"])
        starts = np.asarray(payload["window_start_seconds"])
        for name, array in (("recording_id", recording), ("group_id", group)):
            if array.shape != (x.shape[0],):
                raise ValueError(f"{name} must have one value per window")
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{name} must be an integer array")
            if (array < 0).any():
                raise ValueError(f"{name} cannot contain negative values")
        if starts.shape != (x.shape[0],):
            raise ValueError("window_start_seconds must have one value per window")
        if not np.issubdtype(starts.dtype, np.floating):
            raise ValueError("window_start_seconds must be a floating-point array")
        if not np.isfinite(starts).all():
            raise ValueError("window_start_seconds contains NaN or infinite values")
        if (starts < 0).any():
            raise ValueError("window_start_seconds cannot contain negative values")


def patient_level_split(
    patient_ids: NDArray[np.integer],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 17,
) -> PatientSplit:
    """Split whole patients, never windows, into train/validation/test sets."""

    if isinstance(validation_fraction, (bool, np.bool_)) or isinstance(
        test_fraction, (bool, np.bool_)
    ):
        raise ValueError("split fractions must be real numbers")
    try:
        validation_fraction = float(validation_fraction)
        test_fraction = float(test_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("split fractions must be real numbers") from exc
    if not np.isfinite(validation_fraction) or not np.isfinite(test_fraction):
        raise ValueError("split fractions must be finite")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    raw_ids = np.asarray(patient_ids)
    if raw_ids.ndim != 1:
        raise ValueError("patient_ids must be one-dimensional")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("patient_ids must be an integer array")
    if (raw_ids < 0).any():
        raise ValueError("patient_ids cannot contain negative values")
    if np.issubdtype(raw_ids.dtype, np.unsignedinteger) and raw_ids.size:
        if raw_ids.max() > np.iinfo(np.int64).max:
            raise ValueError("patient_ids cannot exceed the signed 64-bit integer range")
    ids = raw_ids.astype(np.int64, copy=False)
    unique = np.unique(ids)
    requested_partitions = 1 + int(validation_fraction > 0) + int(test_fraction > 0)
    if unique.size < requested_partitions:
        raise ValueError(
            f"at least {requested_partitions} patients are required for the requested split"
        )
    shuffled = np.random.default_rng(seed).permutation(unique)
    test_count = max(1, round(unique.size * test_fraction)) if test_fraction > 0 else 0
    validation_count = (
        max(1, round(unique.size * validation_fraction)) if validation_fraction > 0 else 0
    )
    minimum_test = int(test_fraction > 0)
    minimum_validation = int(validation_fraction > 0)
    while test_count + validation_count >= unique.size:
        if test_count > minimum_test and test_count >= validation_count:
            test_count -= 1
        elif validation_count > minimum_validation:
            validation_count -= 1
        elif test_count > minimum_test:
            test_count -= 1
        else:  # Defensive: the requested-partition check above should make this unreachable.
            raise ValueError("not enough patients to retain a non-empty training partition")
    test_patients = shuffled[:test_count]
    validation_patients = shuffled[test_count : test_count + validation_count]
    train_patients = shuffled[test_count + validation_count :]
    train_mask = np.isin(ids, train_patients)
    validation_mask = np.isin(ids, validation_patients)
    test_mask = np.isin(ids, test_patients)
    return PatientSplit(
        np.flatnonzero(train_mask),
        np.flatnonzero(validation_mask),
        np.flatnonzero(test_mask),
        tuple(int(value) for value in train_patients),
        tuple(int(value) for value in validation_patients),
        tuple(int(value) for value in test_patients),
    )


def remap_patient_ids(patient_ids: NDArray[np.integer]) -> tuple[NDArray[np.int64], dict[int, int]]:
    unique = np.unique(patient_ids)
    mapping = {int(original): index for index, original in enumerate(unique)}
    remapped = np.asarray([mapping[int(value)] for value in patient_ids], dtype=np.int64)
    return remapped, mapping


def patient_episodes(
    x: Tensor,
    y: Tensor,
    patient_ids: Tensor,
    support_per_class: int,
    query_per_class: int,
    seed: int = 17,
    group_ids: Tensor | None = None,
    skip_ineligible: bool = False,
) -> Iterator[Episode]:
    """Yield balanced patient episodes with disjoint recording/temporal groups.

    ``group_ids`` must identify recording sessions or non-overlapping temporal
    blocks. Support and query examples are drawn from disjoint groups. Synthetic
    callers with independently generated windows may use one unique group per
    window; real overlapping windows must never do so.
    """

    if support_per_class < 1 or query_per_class < 1:
        raise ValueError("support_per_class and query_per_class must be positive")
    if x.shape[0] != y.shape[0] or x.shape[0] != patient_ids.shape[0]:
        raise ValueError("x, y, and patient_ids must contain the same number of windows")
    if group_ids is None:
        raise ValueError("group_ids is required to keep recordings or temporal blocks disjoint")
    if group_ids.shape != patient_ids.shape:
        raise ValueError("group_ids must have one value per window")
    generator = torch.Generator().manual_seed(seed)
    for patient in torch.unique(patient_ids).tolist():
        patient_mask = patient_ids == patient
        patient_groups = torch.unique(group_ids[patient_mask])
        if patient_groups.numel() < 2:
            if skip_ineligible:
                continue
            raise ValueError(f"patient {patient} has fewer than two support/query groups")
        chosen_support: dict[int, Tensor] | None = None
        chosen_query: dict[int, Tensor] | None = None
        attempts = max(32, int(patient_groups.numel()) * 4)
        for _ in range(attempts):
            shuffled_groups = patient_groups[
                torch.randperm(patient_groups.numel(), generator=generator)
            ]
            for split_index in range(1, shuffled_groups.numel()):
                support_groups = shuffled_groups[:split_index]
                query_groups = shuffled_groups[split_index:]
                support_mask = patient_mask & torch.isin(group_ids, support_groups)
                query_mask = patient_mask & torch.isin(group_ids, query_groups)
                support_by_label: dict[int, Tensor] = {}
                query_by_label: dict[int, Tensor] = {}
                feasible = True
                for label in (0, 1):
                    support_candidates = torch.nonzero(
                        support_mask & (y.long() == label), as_tuple=False
                    ).flatten()
                    query_candidates = torch.nonzero(
                        query_mask & (y.long() == label), as_tuple=False
                    ).flatten()
                    if (
                        support_candidates.numel() < support_per_class
                        or query_candidates.numel() < query_per_class
                    ):
                        feasible = False
                        break
                    support_by_label[label] = support_candidates[
                        torch.randperm(support_candidates.numel(), generator=generator)[
                            :support_per_class
                        ]
                    ]
                    query_by_label[label] = query_candidates[
                        torch.randperm(query_candidates.numel(), generator=generator)[
                            :query_per_class
                        ]
                    ]
                if feasible:
                    chosen_support = support_by_label
                    chosen_query = query_by_label
                    break
            if chosen_support is not None:
                break
        if chosen_support is None or chosen_query is None:
            if skip_ineligible:
                continue
            raise ValueError(
                f"patient {patient} cannot satisfy the requested class-balanced grouped episode"
            )
        support_indices = torch.cat([chosen_support[label] for label in (0, 1)])
        query_indices = torch.cat([chosen_query[label] for label in (0, 1)])
        yield Episode(
            x[support_indices],
            y[support_indices],
            x[query_indices],
            y[query_indices],
            int(patient),
        )


def synthetic_eeg_windows(
    patients: int = 6,
    windows_per_patient: int = 24,
    channels: int = 8,
    samples: int = 256,
    sampling_rate_hz: float = 128.0,
    seed: int = 17,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """Create non-clinical EEG-like windows for tests and the quickstart."""

    if patients < 2 or windows_per_patient < 4:
        raise ValueError("synthetic data needs at least two patients and four windows each")
    if channels < 1 or samples < 8 or sampling_rate_hz <= 0:
        raise ValueError("channels/rate must be positive and samples must be at least eight")
    rng = np.random.default_rng(seed)
    total = patients * windows_per_patient
    x = np.empty((total, channels, samples), dtype=np.float32)
    y = np.empty(total, dtype=np.int64)
    patient_ids = np.empty(total, dtype=np.int64)
    time = np.arange(samples, dtype=np.float32) / sampling_rate_hz
    index = 0
    for patient in range(patients):
        baseline_frequency = 7.0 + 0.35 * patient
        spatial_signature = rng.normal(0.0, 0.35, size=(channels, 1))
        for window in range(windows_per_patient):
            label = window % 2
            background = 0.35 * rng.normal(size=(channels, samples))
            background += 0.25 * np.sin(2 * np.pi * baseline_frequency * time + spatial_signature)
            if label:
                envelope = np.exp(-0.5 * ((time - time.mean()) / 0.32) ** 2)
                seizure = np.sin(2 * np.pi * (12.0 + 0.2 * patient) * time)
                channel_weights = np.linspace(0.4, 1.2, channels)[:, None]
                background += 1.1 * channel_weights * seizure * envelope
            x[index] = background.astype(np.float32)
            y[index] = label
            patient_ids[index] = patient
            index += 1
    return x, y, patient_ids
