# Dataset preparation

REALM intentionally does not download, bundle, or sublicense EEG recordings. Obtain each
dataset from its official host, accept its current terms, preserve its required citation, and
build local adapters that emit the neutral NPZ contract described below.

## Datasets named in the manuscript

| Dataset | Official source | Access and release note |
|---|---|---|
| CHB-MIT Scalp EEG Database v1.0.0 | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) | Open access under the data license stated on PhysioNet. The original collection has 23 cases from 22 pediatric subjects at 256 Hz; the v1.0.0 file tree also includes added case `chb24`. Treat `chb01` and `chb21` as the same person when splitting. |
| TUH EEG Seizure Corpus (TUSZ) v2.0.6 | [Neural Engineering Data Consortium](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml) | The official page currently lists v2.0.6 and requires an approved access form followed by SSH-key/rsync access. Do not redistribute credentials or corpus files; recheck the current release and terms before use. |
| Siena Scalp EEG Database v1.0.0 | [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) | Open access under CC BY 4.0 as stated by PhysioNet; 14 patients at 512 Hz. Preserve the dataset citation and distinguish EEG from included EKG channels. |

Dataset facts and terms can change. Recheck the official page immediately before acquisition
or redistribution; this repository's Apache-2.0/CC BY 4.0 licenses do not override a dataset's
own terms.

## Neutral NPZ contract

Save one compressed file containing only numeric arrays:

```python
import numpy as np

np.savez_compressed(
    "windows.npz",
    x=x.astype(np.float32),          # [N, C, T]
    y=y.astype(np.int64),            # [N], values 0 or 1
    patient_id=patient_id.astype(np.int64),  # [N]
    # Optional aligned metadata: include all three arrays or none of them.
    recording_id=recording_id.astype(np.int64),       # [N], non-negative
    group_id=group_id.astype(np.int64),               # [N], non-negative
    window_start_seconds=window_start_seconds.astype(np.float64),  # [N]
)
```

Run `realm-eeg validate-npz windows.npz` before training. The validator rejects missing keys,
object arrays, non-finite EEG values, non-binary labels, and negative or non-integer patient
IDs. When temporal metadata is supplied, `recording_id`, `group_id`, and
`window_start_seconds` are an all-or-none set with one value per window. Recording and group
IDs must be non-negative integers; window starts must be non-negative, finite floating-point
seconds. Use `group_id` for recording sessions or non-overlapping temporal blocks suitable
for support/query separation.

Compressed NPZ members cannot be memory-mapped and may exhaust memory at clinical scale.
For large local datasets, write the same arrays as `x.npy`, `y.npy`, and `patient_id.npy` in
one directory and load them with `NPYDirectoryDataset`. If temporal metadata is used, also
write all three of `recording_id.npy`, `group_id.npy`, and `window_start_seconds.npy`. It
memory-maps the arrays and checks EEG finiteness in bounded chunks. The NPZ format is intended
for interchange, tests, and modest datasets.

Patient identifiers may be arbitrary non-negative integers in the interchange file. Before Stage 2, run
`remap_patient_ids` on the **source-train partition only** so its patient labels are contiguous
from zero to `patient_classes - 1`; preserve that non-identifying mapping in the private run
record. Do not add final-test patients to the Stage 2 identity head.

Keep richer provenance in a separate, access-controlled manifest keyed by a non-identifying
window ID. Recommended fields include dataset version, subject key, recording key, source
sample range, montage, original sample rate, preprocessing version, label source, adjudication
status, and the cryptographic hash of the source file. Do not put protected information in the
public NPZ.

## Harmonization pipeline

The reference helper in `realm_eeg.preprocessing` performs:

1. channel-name normalization and mapping to a fixed 19-channel International 10–20 order;
2. zero filling plus a Boolean mask for missing channels;
3. anti-aliased polyphase downsampling (SciPy) or linear upsampling to the configured rate;
4. optional zero-phase bandpass and line-noise notch filtering; and
5. per-window, per-channel z-score normalization.

These are reference choices because the manuscript does not define exact filters, montage,
window duration, overlap, artifact rejection, or reference scheme. Record any changes in a
versioned configuration. Bipolar montage reconstruction and rereferencing are **not**
automatic: convert bipolar derivations to the intended referential channels before calling
`preprocess_window`. The helper requires at least 50% target-channel coverage by default;
change `minimum_channel_fraction` only under a prespecified dataset policy. Retain and audit
the zero-filled channel mask even though the current reference backbone does not consume it
directly.

## Leakage-safe partitions

Split unique people before windowing, balancing, augmentation, normalization-statistic
fitting, or hyperparameter tuning:

```text
source/meta-train patients    Stages 1–2 fitting and Stage 3 outer-loop tasks
validation patients           model selection and early stopping only
final test patients           locked evaluation; support/query split within patient
```

Alternatively, declare a separate patient-disjoint meta-train cohort. In either design,
validation and final-test patients must never contribute an outer-loop update.

For a final test patient, the support examples used for Stage 3 must be separated from query
examples by recording session or non-overlapping temporal block. Pass those block identifiers
as `group_ids` to `patient_episodes`; the sampler refuses an ungrouped split. If multiple
dataset case IDs refer to one person, collapse them to one patient key before splitting. Fit
any external patient-ID probe on a separate partition of frozen embeddings.

The sampler raises when a patient cannot satisfy the requested grouped, class-balanced
support/query episode. An experiment may set `skip_ineligible=True` only with a prespecified
eligibility rule and a reported list/count of excluded patients; otherwise silent cohort
selection can bias the evaluation.

## Label and event policy

Document how sample-level annotations become windows and how windows become alert events.
At minimum, predefine:

- ictal overlap required for a positive window;
- treatment of preictal/postictal and ambiguous intervals;
- minimum alert duration and gap-merging rule;
- exclusion zones around recording discontinuities;
- class sampling/weighting;
- handling of unlabeled time and inter-rater disagreement; and
- whether `K` means total support windows, positive windows, or examples per class.

The reference episode sampler interprets `support_per_class` literally. That convention is an
implementation decision, not a resolved statement from the manuscript.
