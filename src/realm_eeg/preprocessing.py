"""Dataset-independent EEG harmonization and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

STANDARD_1020_19 = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
)

_CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}


@dataclass(frozen=True)
class PreprocessedWindow:
    values: NDArray[np.float32]
    channel_mask: NDArray[np.bool_]
    channel_names: tuple[str, ...]
    sampling_rate_hz: float


def canonical_channel_name(name: str) -> str:
    cleaned = name.upper().replace("EEG", "").replace("-REF", "").replace("-LE", "")
    cleaned = cleaned.replace(" ", "").replace(".", "")
    title = cleaned[:1].upper() + cleaned[1:].lower()
    return _CHANNEL_ALIASES.get(title.upper(), title)


def harmonize_channels(
    values: NDArray[np.floating],
    source_names: Sequence[str],
    target_names: Sequence[str] = STANDARD_1020_19,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Map channels to a fixed order and zero-fill missing channels."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("values must have shape [channels, samples]")
    if len(source_names) != array.shape[0]:
        raise ValueError("source_names length must match the channel axis")
    canonical_sources = [canonical_channel_name(name) for name in source_names]
    duplicates = sorted(
        name for name in set(canonical_sources) if canonical_sources.count(name) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate channels after alias normalization: {duplicates}")
    canonical_targets = [canonical_channel_name(name) for name in target_names]
    if len(set(canonical_targets)) != len(canonical_targets):
        raise ValueError("target_names contains duplicate canonical channels")
    lookup = {name: index for index, name in enumerate(canonical_sources)}
    output = np.zeros((len(target_names), array.shape[1]), dtype=np.float32)
    mask = np.zeros(len(target_names), dtype=np.bool_)
    for target_index, target_name in enumerate(target_names):
        source_index = lookup.get(canonical_channel_name(target_name))
        if source_index is not None:
            output[target_index] = array[source_index].astype(np.float32, copy=False)
            mask[target_index] = True
    return output, mask


def resample_linear(
    values: NDArray[np.floating],
    source_rate_hz: float,
    target_rate_hz: float,
) -> NDArray[np.float32]:
    """Resample safely: polyphase anti-aliasing down, linear interpolation up.

    Downsampling requires the optional SciPy dependency because naïve decimation
    can alias high-frequency content into the target band.
    """

    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sampling rates must be positive")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] < 1:
        raise ValueError("values must have shape [channels, samples] with at least one sample")
    if source_rate_hz == target_rate_hz:
        return array.copy()
    output_samples = round(array.shape[-1] * target_rate_hz / source_rate_hz)
    if target_rate_hz < source_rate_hz:
        try:
            from scipy import signal
        except ImportError as exc:
            raise ImportError(
                "Anti-aliased downsampling requires scipy; install realm-eeg[eeg]"
            ) from exc
        ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(1000)
        output = signal.resample_poly(
            array,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=-1,
            padtype="line",
        ).astype(np.float32)
        if output.shape[-1] > output_samples:
            output = output[..., :output_samples]
        elif output.shape[-1] < output_samples:
            output = np.pad(
                output,
                ((0, 0), (0, output_samples - output.shape[-1])),
                mode="edge",
            )
        return output
    source_time = np.arange(array.shape[-1], dtype=np.float64) / source_rate_hz
    target_time = np.arange(output_samples, dtype=np.float64) / target_rate_hz
    output = np.vstack([np.interp(target_time, source_time, channel) for channel in array]).astype(
        np.float32
    )
    return output


def bandpass_notch_filter(
    values: NDArray[np.floating],
    sampling_rate_hz: float,
    bandpass_hz: tuple[float, float] = (0.5, 45.0),
    notch_hz: float | None = 60.0,
) -> NDArray[np.float32]:
    """Zero-phase Butterworth bandpass and optional line-noise notch filtering."""

    try:
        from scipy import signal
    except ImportError as exc:
        raise ImportError("Filtering requires scipy; install realm-eeg[eeg]") from exc
    low, high = bandpass_hz
    nyquist = sampling_rate_hz / 2.0
    if not 0 < low < high < nyquist:
        raise ValueError("bandpass_hz must satisfy 0 < low < high < Nyquist")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sampling_rate_hz, output="sos")
    output = signal.sosfiltfilt(sos, np.asarray(values), axis=-1)
    if notch_hz is not None:
        if not 0 < notch_hz < nyquist:
            raise ValueError("notch_hz must be below Nyquist")
        b, a = signal.iirnotch(notch_hz, Q=30.0, fs=sampling_rate_hz)
        output = signal.filtfilt(b, a, output, axis=-1)
    return output.astype(np.float32)


def channelwise_zscore(
    values: NDArray[np.floating],
    channel_mask: NDArray[np.bool_] | None = None,
    epsilon: float = 1e-6,
) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [channels, samples]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if channel_mask is not None and np.asarray(channel_mask).shape != (array.shape[0],):
        raise ValueError("channel_mask must have one value per channel")
    means = array.mean(axis=-1, keepdims=True)
    standard_deviations = array.std(axis=-1, keepdims=True)
    normalized = (array - means) / np.maximum(standard_deviations, epsilon)
    if channel_mask is not None:
        normalized = normalized * np.asarray(channel_mask, dtype=np.float32)[:, None]
    return normalized.astype(np.float32)


def segment_signal(
    values: NDArray[np.floating],
    window_samples: int,
    overlap_samples: int = 0,
) -> NDArray[np.float32]:
    if window_samples < 1:
        raise ValueError("window_samples must be positive")
    if overlap_samples < 0:
        raise ValueError("overlap_samples cannot be negative")
    step = window_samples - overlap_samples
    if step < 1:
        raise ValueError("overlap_samples must be smaller than window_samples")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [channels, samples]")
    starts = range(0, array.shape[-1] - window_samples + 1, step)
    windows = [array[:, start : start + window_samples] for start in starts]
    if not windows:
        return np.empty((0, array.shape[0], window_samples), dtype=np.float32)
    return np.stack(windows)


def preprocess_window(
    values: NDArray[np.floating],
    source_names: Sequence[str],
    source_rate_hz: float,
    target_rate_hz: float = 256.0,
    target_names: Sequence[str] = STANDARD_1020_19,
    apply_filter: bool = False,
    notch_hz: float | None = 60.0,
    minimum_channel_fraction: float = 0.5,
) -> PreprocessedWindow:
    """Harmonize and normalize one window after enforcing channel coverage.

    Bipolar montage reconstruction and rereferencing are not automatic. Callers
    must convert such signals to the expected referential channels before this
    helper and audit the returned channel mask.
    """

    if isinstance(minimum_channel_fraction, (bool, np.bool_)):
        raise ValueError("minimum_channel_fraction must satisfy 0 < value <= 1")
    try:
        minimum_channel_fraction = float(minimum_channel_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_channel_fraction must satisfy 0 < value <= 1") from exc
    if not np.isfinite(minimum_channel_fraction) or not 0 < minimum_channel_fraction <= 1:
        raise ValueError("minimum_channel_fraction must satisfy 0 < value <= 1")
    harmonized, mask = harmonize_channels(values, source_names, target_names)
    channel_fraction = float(mask.mean()) if mask.size else 0.0
    if channel_fraction < minimum_channel_fraction:
        raise ValueError(
            f"channel coverage {channel_fraction:.1%} is below the required "
            f"{minimum_channel_fraction:.1%}; bipolar montage reconstruction and "
            "rereferencing are not automatic"
        )
    resampled = resample_linear(harmonized, source_rate_hz, target_rate_hz)
    if apply_filter:
        resampled = bandpass_notch_filter(resampled, target_rate_hz, notch_hz=notch_hz)
    normalized = channelwise_zscore(resampled, mask)
    return PreprocessedWindow(normalized, mask, tuple(target_names), target_rate_hz)
