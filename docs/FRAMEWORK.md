# REALM framework

> **Status:** REALM is a proposed research framework. This repository is an
> unvalidated reference implementation, not a clinically validated model or a
> medical device.

REALM expands to **Robust EEG Adaptation via meta-Learning and disentanglement
Modules**. It is a four-stage framework for cross-patient EEG seizure detection:
population pretraining, patient-identity disentanglement, few-shot adaptation,
and prediction-level explanation. REALM is the canonical public name used in
the final framework manuscript.

## Purpose

The framework treats each patient as a domain. Its central hypothesis is that a
model trained only for seizure classification can organize its latent space
around patient identity rather than pathology. REALM therefore trains a shared
CNN–Transformer backbone in stages so that seizure/non-seizure structure becomes
dominant before patient-specific adaptation.

The intended output is physician-facing decision support: seizure probabilities,
candidate intervals, and auditable explanation artifacts. A clinician remains
responsible for accepting, rejecting, or editing any suggested event.

## Notation and canonical equations

An EEG segment is

$$
x \in \mathbb{R}^{C \times T},
$$

where $C$ is the number of EEG channels and $T$ is the number of temporal
samples. Its seizure label and patient identity are

$$
y \in \{0,1\}, \qquad
p \in \{1,\ldots,N_{\mathrm{pat}}\},
$$

with $y=1$ denoting ictal activity and $y=0$ denoting non-seizure activity. The
shared backbone and latent representation are

$$
B_\theta, \qquad
z = B_\theta(x), \qquad
z \in \mathbb{R}^{d}.
$$

The manuscript writes the seizure probability as

$$
p_i = \sigma\!\left(f_{\mathrm{seiz}}\!\left(B_\theta(x_i)\right)\right),
$$

where $f_{\mathrm{seiz}}$ is the seizure head and $\sigma$ is the sigmoid
function. The symbol $p$ is overloaded in the manuscript for patient identity
and $p_i$ for seizure probability; the implementation uses distinct variable
names.

### Stage 1 loss

The canonical per-example binary cross-entropy is

$$
\mathcal{L}_{\mathrm{seiz}}
= -\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right].
$$

The manuscript does not prescribe batch reduction, class weighting, or a
negative-sampling strategy. The reference implementation uses mean binary
cross-entropy with logits.

### Stage 2 objective

The representation goal is to reduce patient information while retaining
seizure information:

$$
\min I(z;p)
\quad\text{while preserving}\quad
I(z;y).
$$

The gradient-reversal layer (GRL) is the identity during the forward pass,

$$
\operatorname{GRL}(z)=z,
$$

and multiplies the gradient entering the backbone by $-\lambda$ during
backpropagation. The manuscript summarizes the backbone's adversarial objective
as

$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{seiz}} - \lambda\mathcal{L}_{\mathrm{pat}}.
$$

Here $\mathcal{L}_{\mathrm{pat}}$ is patient-identity cross-entropy and
$\lambda$ controls the adversarial pressure.

#### Safe GRL loss semantics

The negative expression above is a minimax/backbone-gradient description. It
must not be minimized naively over the patient-head parameters: doing so would
teach the patient classifier to become worse. With an autograd GRL, the safe
implementation is

$$
\mathcal{L}_{\mathrm{optimizer}}
= \mathcal{L}_{\mathrm{seiz}} + w_{\mathrm{pat}}\mathcal{L}_{\mathrm{pat}},
$$

while the GRL applies

$$
\frac{\partial \mathcal{L}_{\mathrm{pat}}}{\partial z}
\longmapsto
-\lambda
\frac{\partial \mathcal{L}_{\mathrm{pat}}}{\partial z}.
$$

Consequently, the patient head minimizes ordinary positive cross-entropy, the
seizure head minimizes seizure loss, and the backbone receives the reversed,
scaled patient gradient. This is how `stage2_loss` and `gradient_reverse` are
implemented in this repository. The coefficient follows a DANN-style smooth
schedule from approximately zero toward one; the exact schedule is an
implementation choice because the manuscript specifies only that $\lambda$
increase gradually from 0 to 1.

### Stage 3 adaptation

For a patient task, the support set and inner update are

$$
S = \{(x_i,y_i)\}_{i=1}^{K},
$$

$$
\theta' = \theta - \alpha\nabla_\theta\mathcal{L}_{S}(\theta),
$$

where $\alpha$ is the inner-loop learning rate. With a disjoint query set $Q_i$
for task $T_i$, the MAML objective is

$$
\min_\theta \sum_{T_i}\mathcal{L}_{Q_i}(\theta'_i).
$$

The experimental protocol considers

$$
K \in \{1,5,10,20\}.
$$

The manuscript defines $\theta$ as backbone parameters but does not resolve
whether the seizure head is also adapted, how many inner steps are used, or
whether $K$ is total support size, positive examples, or examples per class.
The reference default makes the following explicit choices:

- first-order MAML;
- one differentiable inner step during meta-training;
- adaptation of both the backbone and seizure head;
- exclusion of the patient head from Stage 3;
- balanced episode construction, with the configured support count interpreted
  **per class** by `patient_episodes`;
- recording/session or non-overlapping temporal-group separation between each
  support and query set via the required `group_ids` argument;
- deployment adaptation on a copy of the model, leaving the source checkpoint
  unchanged.

These are reproducible reference defaults, not empirically established optimal
settings. Experiments should report whether $K$ means per-class or total examples
and should include a head-only adaptation ablation.

### Combined objective

The final DOCX's combined equation is missing because of a document-formatting
loss, although its surrounding definitions and the preceding manuscript version
preserve the intended expression:

$$
\mathcal{L}_{\mathrm{REALM}}
= \mathcal{L}_{\mathrm{seiz}}
- \lambda\mathcal{L}_{\mathrm{pat}}
+ \beta\mathcal{L}_{\mathrm{meta}}.
$$

This equation is best read as a summary of the three learning pressures. REALM
is described as a **sequential** pipeline, so the reference implementation starts
Stage 3 from a Stage 2 checkpoint rather than treating the expression as one
monolithic optimizer loss. The reference workflow also resets AdamW state at each
stage boundary; its portable checkpoints contain model weights, configuration, and
safe provenance metadata but not optimizer moments.

## Four-stage workflow

### Stage 1 — population-level pretraining

The backbone is trained from scratch on multi-patient EEG windows using seizure
labels. The CNN captures local temporal/spectral patterns, and the Transformer
captures longer temporal relationships. Stage 1 intentionally has no invariance
constraint, so its embeddings may retain patient identity.

Inputs:

- harmonized EEG windows shaped `[batch, channels, samples]`;
- binary seizure labels;
- patient identifiers retained for splitting and later adversarial training.

Outputs:

- a population-pretrained backbone and seizure head;
- latent embeddings for geometry diagnostics.

### Stage 2 — domain-adversarial disentanglement

The Stage 1 weights are continued with a patient-identity head connected through
the GRL. The manuscript describes three qualitative phases:

1. Patient clusters remain intact early in training.
2. Cluster boundaries blur during a temporary "confusion valley."
3. Patient clusters dissolve and seizure/non-seizure structure becomes dominant.

The intended verification is twofold: visualize frozen embeddings with t-SNE or
UMAP, and train a separate held-out patient-identity probe. The probe should be
near an appropriately defined chance baseline while seizure discrimination is
preserved. A plot alone is not sufficient evidence of invariance.

`patient_identity_linear_probe` fits a deterministic multiclass linear probe to
precomputed, detached embeddings using caller-supplied, disjoint train and test
indices. It reports held-out accuracy, the test accuracy of always predicting
the most frequent training-split patient (ties use the smallest patient ID), and
the expected accuracy of a uniform random guess over the training patient
classes. Every test patient class must be represented in the probe-training
split. Split probe examples by recording/session or non-overlapping temporal
group before passing indices; randomly splitting overlapping windows would leak
patient-specific signal.

Near-baseline linear-probe accuracy is only a diagnostic result for this probe
capacity and split. It does not prove patient invariance, rule out nonlinear
identity recovery, or support a privacy guarantee.

### Stage 3 — few-shot meta-learning and adaptation

Patients are treated as tasks. Meta-training adapts on each support set and
optimizes the initialization with the corresponding disjoint query loss. At
deployment, the resulting model is copied and adapted on a new patient's labeled
support set before evaluating unseen windows from that patient.

The reference implementation does not claim true zero-shot personalization. If
no support labels exist, use the generalized Stage 2/meta-learned model and label
the evaluation as zero-shot rather than few-shot.

### Stage 4 — interpretation and accountability

Stage 4 does not change model predictions. It can produce:

- SHAP attribution when the optional SHAP dependency and a background set are
  supplied;
- integrated gradients as a lightweight attribution fallback;
- residual-aware Transformer attention rollout;
- Euclidean distance to seizure and non-seizure latent centroids.

Attribution, attention, and centroid proximity are audit aids, not proof of
causality, calibration, or clinical correctness. Audit examples should include
artifact, movement, muscle activity, drift, and disconnected-channel failure
cases, not only visually favorable seizure examples.

## Strict patient separation

Patient-independent evaluation is non-negotiable. The manuscript states both

$$
P_{\mathrm{train}}(X,Y) \ne P_{\mathrm{test}}(X,Y)
$$

and

$$
P_{\mathrm{train}} \cap P_{\mathrm{test}} = \varnothing.
$$

The final DOCX drops the first expression's equation object during formatting;
the preceding manuscript version preserves the displayed form transcribed here.
The disjoint-patient constraint remains visible in the final DOCX itself.

All windows from one patient must remain in one patient partition. Never split
windows first and then assign them to train and test sets.

MAML introduces another leakage boundary that the manuscript leaves ambiguous.
Use the following safe protocol:

1. Create patient-disjoint source-train, validation, and final-test partitions.
2. Run Stage 1 and Stage 2 only on source-train patients.
3. Meta-train Stage 3 on source-train patients or on a separately declared
   meta-train patient set.
4. Never use a final-test patient's query windows in an outer-loop update,
   hyperparameter selection, early stopping, normalization fit, centroid fit, or
   background-set construction.
5. For each final-test patient, create disjoint support and query windows. Adapt
   only on support; report performance only on query.

Equivalently, the minimum required conditions are

$$
P_{\mathrm{stage1/2}} \cap P_{\mathrm{final\ test}} = \varnothing,
\qquad
P_{\mathrm{meta\ train}} \cap P_{\mathrm{final\ test}} = \varnothing,
\qquad
S_j \cap Q_j = \varnothing.
$$

If a dedicated adaptation cohort is held out from Stages 1 and 2, split that
cohort again into meta-train/meta-validation/final-meta-test patients before any
outer-loop optimization.

## Data and preprocessing contract

The repository's portable interchange format requires:

- `x`: finite floating-point array `[windows, channels, samples]`;
- `y`: binary array `[windows]`;
- `patient_id`: integer array `[windows]`.
- optional, all-or-none `recording_id`, `group_id`, and
  `window_start_seconds` arrays, each shaped `[windows]`, for grouped episodes
  and continuous-timeline alert metrics.

The manuscript calls for channel-name harmonization, montage reconstruction,
sampling-rate harmonization, bandpass and notch filtering, normalization,
artifact handling, and fixed-window segmentation. It proposes the maximal common
set of approximately 19–21 standard 10–20 scalp channels and requires tolerance
of missing/reordered channels. Exact filter cutoffs, window duration, overlap,
artifact method, and resampling target are not prescribed by the framework;
repository defaults must therefore be reported as implementation choices.

Clinical metadata is not a required backbone input. The manuscript suggests that
future metadata, if used, should be a separate calibration or adjudication layer
to reduce institution-specific shortcuts.

## Evaluation contract

Canonical metrics include

$$
\mathrm{Sensitivity} = \frac{TP}{TP+FN},
\qquad
\mathrm{Specificity} = \frac{TN}{TN+FP},
$$

plus precision, F1, and ROC AUC. Alert burden is

$$
\mathrm{False\ alarms\ per\ hour}
= \frac{\mathrm{False\ alarms}}{\mathrm{recording\ hours}}.
$$

The manuscript labels the last quantity "FDR," but this repository uses the
unambiguous name `false_alarm_events_per_hour` because FDR commonly means false
discovery rate.

Required comparisons are a patient-specific model, ordinary transfer learning
without Stage 2, the Stage 1 backbone alone, and component ablations. Report
patient-level uncertainty, not confidence intervals formed by treating windows
as independent patients. The proposed protocol includes leave-one-patient-out
evaluation, paired comparisons, bootstrap confidence intervals, and DeLong AUC
testing.

## What remains unspecified

The framework manuscript does not fix architecture depth/width, convolution
kernels, attention heads, positional encoding, class weighting, event-merging
rules, alarm refractory periods, threshold calibration, GRL schedule, MAML order
or inner steps, SHAP background construction, centroid metric, or an uncertainty
calibration method. Repository defaults provide runnable examples but must not be
presented as canonical or clinically optimized.

The manuscript also acknowledges label disagreement and institution/hardware
shift without providing a label-noise model or a site/device adversary. Those
deployment barriers remain open research problems.

## Validation status and limitations

The manuscript contains an experimental protocol, not completed experimental
results. Its feature-space, attribution, and few-shot performance figures are
simulated or illustrative. No sensitivity, specificity, AUC, false-alarm burden,
or external-site performance claim is established by those figures.

Known limitations include computational cost, adversarial instability, possible
removal of clinically meaningful patient-specific structure, dependence on
population diversity, the need for labeled adaptation examples, unresolved
zero-shot behavior, and no validation on wearable, low-channel, intracranial,
subscalp, or real-time streaming systems.

See [TABLES.md](TABLES.md) for the two framework tables and
[MODEL_CARD.md](../MODEL_CARD.md) for the implementation-facing use and risk
statement.
