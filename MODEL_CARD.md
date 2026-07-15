# Model card: REALM EEG reference implementation

## Model details

- **Name:** REALM EEG
- **Expansion:** Robust EEG Adaptation via meta-Learning and disentanglement
  Modules
- **Repository package:** `realm-eeg`
- **Version:** 0.1.0
- **Software license:** Apache-2.0
- **Document license:** CC BY 4.0
- **Development status:** Alpha research reference
- **Validation status:** Unvalidated; no clinical performance claims

REALM is a four-stage framework for cross-patient EEG seizure detection. This
repository implements its population-pretraining, adversarial disentanglement,
few-shot adaptation, and explanation components as a reproducible research
starting point. It does not distribute trained clinical weights or patient data.

## Important notice

This software is **not a medical device**, is **not cleared or approved for
clinical use**, and must not be used to diagnose epilepsy, rule out seizures,
trigger treatment, or replace review by a qualified clinician. The associated
manuscript proposes an experimental protocol; its geometry, attribution, and
few-shot performance figures are simulated or illustrative rather than completed
validation results.

## Intended uses

Appropriate uses include:

- research on patient-independent EEG representation learning;
- reproducible comparison of pretraining, domain-adversarial training, and
  few-shot adaptation;
- development of patient-level data-splitting and evaluation pipelines;
- ablation studies of Stage 1, Stage 2, and Stage 3;
- exploratory auditing with attribution, attention, and latent-space diagnostics;
- synthetic-data smoke tests and educational demonstrations.

Any study using real EEG should obtain the required ethics approval, data-use
authorization, dataset license, and privacy controls.

## Out-of-scope uses

Do not use this implementation for:

- autonomous seizure diagnosis or exclusion;
- bedside alarms, medication changes, emergency escalation, or patient triage;
- retrospective clinical performance claims without external validation;
- surveillance or re-identification of patients;
- training/evaluation splits that place one patient's windows in multiple
  partitions;
- claims that SHAP, integrated gradients, or attention reveal clinical causality;
- wearable, intracranial, subscalp, or real-time deployment without dedicated
  validation for that modality and environment.

## Architecture

The model contains:

1. a one-dimensional CNN feature extractor;
2. Transformer encoder blocks for temporal-token interactions;
3. a pooled latent vector;
4. a binary seizure head;
5. a patient-identity head connected through a gradient-reversal layer during
   Stage 2.

Current runnable defaults are implementation choices, not canonical values from
the framework manuscript:

| Setting | Reference default |
|---|---:|
| Sampling rate | 256 Hz |
| Segment duration | 4 seconds |
| Input channels | 19 |
| CNN width | 64 |
| CNN kernels | 7, 5, 3 |
| Latent dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Positional encoding | Deterministic sinusoidal |
| Feed-forward dimension | 256 |
| Dropout | 0.1 |
| Patient classes | 22 |

Each convolutional block uses convolution, group normalization, GELU, max
pooling, and dropout. The manuscript does not prescribe these numerical values;
they should be tuned and reported for every experiment.

## Inputs and outputs

The NPZ interchange format requires:

- `x`: finite float array `[windows, channels, samples]`;
- `y`: binary array `[windows]`;
- `patient_id`: integer array `[windows]`.

It optionally accepts an all-or-none aligned trio: non-negative integer
`recording_id`, non-negative integer `group_id`, and finite non-negative
`window_start_seconds`. Those fields support recording/temporal-block-separated
episodes and continuous-timeline alert metrics; omitting only part of the trio is
an error.

The default preprocessing helpers can harmonize a 19-channel 10–20 montage,
zero-fill missing channels, resample, optionally apply bandpass/notch filtering,
and perform channelwise z-scoring. `preprocess_window` requires at least 50%
target-channel coverage by default. Bipolar montage reconstruction and
rereferencing are not automatic. A zero-filled channel mask is returned by the
preprocessor, but the current backbone does not consume that mask directly; retain
and audit it, especially when evaluating data with substantial channel missingness.

Primary model outputs are seizure logits/probabilities and latent embeddings.
Stage 2 additionally emits patient logits during training. Optional Stage 4
outputs include input attribution, attention rollout, and distances to
seizure/non-seizure centroids.

## Training stages

### Stage 1: population pretraining

The CNN–Transformer backbone and seizure head minimize mean binary
cross-entropy. Patient identity is not used in the loss, although patient IDs are
required for leak-safe partitioning.

### Stage 2: adversarial disentanglement

The patient head minimizes positive multiclass cross-entropy. The optimizer adds
seizure loss and patient loss, while the GRL reverses and scales only the patient
gradient seen by the backbone. This distinction is essential: directly
minimizing a negative patient loss over the patient-head parameters would be
incorrect.

The default GRL coefficient uses a smooth DANN schedule approaching one. The
framework manuscript requires a gradual 0-to-1 schedule but does not specify its
functional form.

The post-training `patient_identity_linear_probe` diagnostic evaluates frozen
embeddings on explicit, disjoint train/test indices. It reports held-out patient
accuracy together with training-majority and uniform-random baselines. A result
near either baseline is probe- and split-specific evidence only, not proof of
patient invariance or privacy; nonlinear probes and seizure-task preservation
must be evaluated separately.

### Stage 3: meta-learning and patient adaptation

The manuscript defines an inner update for $\theta$ but does not unambiguously
state whether $\theta$ includes the seizure head. It also leaves the MAML order,
inner-step count, and meaning of $K$ unspecified.

Reference meta-training defaults are first-order MAML, one inner step, an inner
learning rate of 0.01, and adaptation of the backbone plus seizure head while
excluding the patient head. Episode construction is balanced and interprets the
support count as examples **per class**, so `support_per_class=K` produces `2K`
total support windows. Studies must report this convention explicitly.

Deployment-time `adapt_to_patient` returns a copied model and, by default, adapts
the backbone and seizure head for five SGD steps at a learning rate of 0.001.
These deployment defaults intentionally differ from the differentiable
meta-training inner-loop defaults and should be validated or aligned in a formal
study. A head-only ablation is strongly recommended.

### Stage 4: explanation

The package supports optional SHAP GradientExplainer attribution, integrated
gradients, residual-aware attention rollout, and Euclidean centroid distance.
The convenience `explain_prediction` path currently uses integrated gradients;
SHAP requires an explicit background set and the optional explanation
dependencies.

Explanation outputs do not alter the seizure score and must not be interpreted
as causal evidence or as a substitute for clinician review. Background data and
centroids must be fitted without final-test leakage.

## Patient-independent data policy

All partitions must be created at the patient level:

$$
P_{\mathrm{train}} \cap P_{\mathrm{test}} = \varnothing.
$$

At minimum:

- Stage 1 and Stage 2 use source-train patients only;
- Stage 3 outer-loop updates never use final-test patients;
- validation patients are used only for tuning and stopping;
- each final-test patient has disjoint support and query windows;
- query windows, test labels, test centroids, and test explanation backgrounds
  are never used during fitting or model selection.

If an adaptation cohort is held out from Stages 1 and 2, it must be split again
into meta-train, meta-validation, and final-meta-test patients before Stage 3.
The provided split helpers are building blocks; the experiment owner remains
responsible for enforcing every stage-specific boundary.

## Evaluation

No benchmark results are bundled or claimed. A valid evaluation should report,
per patient and in aggregate:

- sensitivity;
- specificity;
- precision and F1;
- ROC AUC;
- false alarms per recording hour;
- uncertainty intervals based on patient-level resampling;
- support-set convention and size;
- patient-probe accuracy after Stage 2;
- Stage 1-only, transfer-learning, patient-specific, and component-ablation
  baselines.

Window-level accuracy alone is insufficient. Event-merging logic, thresholds,
minimum duration, refractory intervals, and recording-hour denominators must be
specified before false-alarm results are comparable.

## Data considerations

The framework discusses CHB-MIT, the Temple University EEG Seizure Corpus, and
the Siena Scalp EEG Database. Those datasets are not included. Users are
responsible for access terms, citations, local preprocessing, annotation
interpretation, and any redistribution restrictions.

Expected sources of variation include age and neuroanatomy, seizure type,
medication, electrode placement and impedance, montage, sampling rate, hardware,
institutional workflow, artifacts, and label disagreement. A patient adversary
does not by itself guarantee invariance to site, device, demographic group, or
seizure subtype.

## Limitations and risks

- The framework has not yet established prospective or external-site clinical
  performance.
- Excessive adversarial pressure can remove clinically meaningful individualized
  seizure structure.
- Weak adversarial pressure can leave patient identity encoded in the latent
  representation.
- Patient-probe failure depends on probe capacity and class balance and is not
  proof that all identity information is absent.
- MAML can overfit a small support set and can leak final-test information if
  patient/task boundaries are not explicit.
- Seizure labels can be incomplete or inconsistent; this implementation has no
  dedicated label-noise or adjudication model.
- Class imbalance, thresholding, and window-to-event aggregation materially
  affect false-negative and false-alarm rates.
- t-SNE/UMAP can distort geometry and should not be used as the sole validation
  of disentanglement.
- Attribution and attention can highlight artifacts or unstable features.
- Current defaults have not been established as optimal for any dataset or
  population.
- Computational requirements increase across adversarial and meta-learning
  stages.
- True zero-shot patient personalization remains unresolved.

## Fairness, privacy, and human oversight

Performance should be stratified, where permitted and statistically defensible,
by relevant demographic, clinical, seizure-type, site, and hardware groups.
Underrepresented groups may receive poorer generalization even when aggregate
metrics appear strong.

EEG and patient identifiers are sensitive health data. Use de-identified local
identifiers, minimize retained metadata, restrict access to checkpoints and
embeddings, and do not publish raw records or reversible patient mappings.
Latent embeddings and trained weights should still be treated as potentially
sensitive research artifacts.

Every clinical-facing study should define how clinicians review model outputs,
how disagreements are handled, how failures are logged, and how the system can be
disabled. Human oversight is a required part of the proposed use, not an optional
interface feature.

## Reproducibility reporting

For every experiment, preserve and publish when allowed:

- patient-level split manifests using de-identified IDs;
- preprocessing and channel-mapping configuration;
- architecture and optimizer configuration;
- random seeds and software versions;
- Stage 1, Stage 2, and Stage 3 checkpoint provenance;
- GRL schedule and patient-loss weight;
- MAML order, adapted parameters, inner steps, and support convention;
- threshold and event-aggregation rules;
- patient-level metrics and uncertainty calculations;
- explanation background and centroid provenance.

Further mathematical and workflow detail is in
[docs/FRAMEWORK.md](docs/FRAMEWORK.md), with manuscript tables in
[docs/TABLES.md](docs/TABLES.md).
