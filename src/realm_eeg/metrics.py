"""Patient-independent binary classification and alert-burden metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    accuracy: float
    roc_auc: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def roc_auc(labels: NDArray[np.integer], scores: NDArray[np.floating]) -> float:
    """Compute AUC with average ranks for tied scores."""

    raw_labels = np.asarray(labels)
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("labels must use an integer dtype")
    y = raw_labels.astype(np.int64, copy=False)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.ndim != 1 or y.shape != values.shape or y.size == 0:
        raise ValueError("labels and scores must be aligned, non-empty one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(values).all():
        raise ValueError("AUC requires binary labels and finite scores")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[y == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def binary_metrics(
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    threshold: float = 0.5,
) -> BinaryMetrics:
    raw_labels = np.asarray(labels)
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("labels must use an integer dtype")
    y = raw_labels.astype(np.int64, copy=False)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or y.shape != scores.shape:
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    if y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be a non-empty binary array")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("probabilities must lie between zero and one")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    predicted = scores >= threshold
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = _safe_ratio(tp + tn, y.size)
    return BinaryMetrics(
        sensitivity,
        specificity,
        precision,
        f1,
        accuracy,
        roc_auc(y, scores),
        tp,
        tn,
        fp,
        fn,
    )


def false_alarm_events_per_hour(
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    *,
    recording_ids: NDArray[np.integer],
    window_start_seconds: NDArray[np.floating],
    total_recording_hours: float,
    merge_gap_seconds: float,
    threshold: float = 0.5,
) -> float:
    """Count temporally contiguous false-positive runs per explicit recording hour.

    Windows are sorted by recording and start time. Two false-positive windows
    belong to the same alert only when no intervening non-false window occurs
    and their start-time gap is at most ``merge_gap_seconds``.
    """

    if (
        not np.isfinite(total_recording_hours)
        or not np.isfinite(merge_gap_seconds)
        or total_recording_hours <= 0
        or merge_gap_seconds < 0
    ):
        raise ValueError(
            "recording hours must be finite and positive and merge gap finite and non-negative"
        )
    raw_labels = np.asarray(labels)
    raw_recordings = np.asarray(recording_ids)
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("labels must use an integer dtype")
    if not np.issubdtype(raw_recordings.dtype, np.integer):
        raise ValueError("recording_ids must use an integer dtype")
    y = raw_labels.astype(np.int64, copy=False)
    scores = np.asarray(probabilities, dtype=np.float64)
    recordings = raw_recordings.astype(np.int64, copy=False)
    starts = np.asarray(window_start_seconds, dtype=np.float64)
    if (
        y.ndim != 1
        or scores.ndim != 1
        or recordings.ndim != 1
        or starts.ndim != 1
        or not (y.shape == scores.shape == recordings.shape == starts.shape)
        or y.size == 0
    ):
        raise ValueError(
            "labels, probabilities, recording IDs, and starts must be aligned "
            "one-dimensional arrays"
        )
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and probabilities finite")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("probabilities must lie between zero and one")
    if not np.isfinite(starts).all() or (starts < 0).any():
        raise ValueError("window starts must be finite and non-negative")
    if (recordings < 0).any():
        raise ValueError("recording_ids must be non-negative")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    pairs = list(zip(recordings.tolist(), starts.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("each recording/start-time pair must be unique")
    try:
        order = np.lexsort((starts, recordings))
    except TypeError as exc:
        raise ValueError("recording_ids must be sortable") from exc
    y = y[order]
    predicted = scores[order] >= threshold
    recordings = recordings[order]
    starts = starts[order]
    false_alarms = 0
    previous_false = False
    previous_recording = None
    previous_start = 0.0
    for truth, prediction, recording, start in zip(y, predicted, recordings, starts):
        current_false = bool(prediction and truth == 0)
        same_run = (
            previous_false
            and recording == previous_recording
            and start - previous_start <= merge_gap_seconds
        )
        if current_false and not same_run:
            false_alarms += 1
        previous_false = current_false
        previous_recording = recording
        previous_start = float(start)
    return false_alarms / total_recording_hours


def bootstrap_interval(
    values: NDArray[np.floating],
    statistic=np.mean,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 17,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("at least two observations in a one-dimensional array are required")
    if not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be in (0, 1) and resamples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [statistic(rng.choice(array, size=array.size, replace=True)) for _ in range(resamples)]
    )
    if estimates.ndim != 1 or not np.isfinite(estimates).all():
        raise ValueError("statistic must return one finite scalar per resample")
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))
