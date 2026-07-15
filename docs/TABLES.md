# Framework tables

These tables are normalized transcriptions of the two tables in the canonical
REALM framework manuscript. They describe the proposed design; they are not
empirical performance results.

Machine-readable copies:

- [Table 1 CSV](assets/tables/table-01-backbone-role-by-stage.csv)
- [Table 2 CSV](assets/tables/table-02-transparency-profile.csv)

## Table 1. The backbone's role across all four REALM stages

| Stage | Backbone role | Objective |
|---|---|---|
| Stage 1 — Population Pretraining | Trained from scratch on diverse multi-patient data | Learn broad seizure-relevant EEG features across populations |
| Stage 2 — Adversarial Disentanglement | Reshaped by adversarial pressure from gradient reversal | Unlearn patient identity; preserve seizure discriminability |
| Stage 3 — Few-Shot Adaptation | Lightly fine-tuned on new patient support set via MAML | Personalize efficiently on top of generalized knowledge |
| Stage 4 — Interpretability | Queried by SHAP, attention visualization, feature positioning | Explain predictions in clinically legible terms |

The same backbone weights are carried forward and progressively refined through
Stages 1–3. Stage 4 queries those weights for explanations and does not update the
model or alter its prediction.

## Table 2. Transparency profile of REALM versus baseline architectures

| Component | Simple CNN | CNN + Transfer | DANN Only | REALM (Ours) |
|---|---|---|---|---|
| Seizure detection loss | ✓ Transparent | ✓ Transparent | ✓ Transparent | ✓ Transparent |
| Learned backbone weights | ✗ Black box | ✗ Black box | ✗ Black box | ✗ Black box |
| Adversarial objective (GRL) | — N/A | — N/A | ✓ Transparent | ✓ Transparent |
| MAML meta-learning objective | — N/A | — N/A | — N/A | ✓ Transparent |
| Backbone constrained adversarial | ✗ No | ✗ No | ✓ Yes | ✓ Yes |
| Feature geometry verifiable | ✗ No | ✗ No | Partial | ✓ Yes (t-SNE) |
| SHAP attribution | ✗ None | ✗ None | ✗ None | ✓ Yes |
| Attention visualization | ✗ None | ✗ None | ✗ None | ✓ Yes |
| Feature space positioning | ✗ None | ✗ None | ✗ None | ✓ Yes |
| Prediction-level explanation | ✗ None | ✗ None | ✗ None | ✓ Yes |

Table 2 expresses the manuscript's transparency argument: learned deep-network
weights remain representationally opaque in every architecture, while REALM adds
explicit mechanisms to constrain, inspect, and explain them. "Verifiable" here
means auditable with diagnostics; it does not mean formally proven, causally
interpretable, calibrated, or clinically validated.

## Interpretation notes

- A GRL is parameter-free, but the backbone and both classifier heads remain
  learned models.
- t-SNE is a visualization, not a quantitative invariance test. Pair it with an
  external patient-identity probe and seizure-performance checks.
- SHAP and attention maps are post-hoc audit aids. Neither establishes that a
  prediction is physiologically causal or clinically correct.
- Stage 4 features are implemented as optional analysis outputs and do not modify
  the seizure score.
