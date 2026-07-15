# REALM Figures

This index is the publication-facing source of truth for REALM figure captions, alternative text, provenance, and downloadable formats. Implementation and QA requirements are defined separately in [FIGURE_CONTRACTS.md](FIGURE_CONTRACTS.md).

## File map

| Figure | Classification | PNG | SVG | Source data | Seed |
|---|---|---|---|---|---|
| 1. Four-stage pipeline | Conceptual | [PNG](assets/figures/figure-01-realm-pipeline.png) | [SVG](assets/figures/figure-01-realm-pipeline.svg) | None | None |
| 2. Gradient-reversal architecture | Conceptual | [PNG](assets/figures/figure-02-gradient-reversal-architecture.png) | [SVG](assets/figures/figure-02-gradient-reversal-architecture.svg) | None | None |
| 3. Feature-space evolution | **Simulated** | [PNG](assets/figures/figure-03-feature-space-evolution-simulated.png) | [SVG](assets/figures/figure-03-feature-space-evolution-simulated.svg) | [CSV](assets/figure-data/figure-03-simulated-embedding.csv) | `31703` |
| 4. EEG attribution | **Simulated** | [PNG](assets/figures/figure-04-shap-eeg-simulated.png) | [SVG](assets/figures/figure-04-shap-eeg-simulated.svg) | [CSV](assets/figure-data/figure-04-simulated-eeg-shap.csv) | `40404` |
| 5. Few-shot performance | **Illustrative** | [PNG](assets/figures/figure-05-few-shot-performance-illustrative.png) | [SVG](assets/figures/figure-05-few-shot-performance-illustrative.svg) | [CSV](assets/figure-data/figure-05-illustrative-performance.csv) | None; fixed hypothetical values |

SVG is preferred for web and typesetting because labels remain scalable and selectable. PNG is supplied as a 300 dpi fallback.

## Figure 1. REALM four-stage pipeline

**Classification:** Conceptual architecture; no measured data.

**Full caption:** REALM four-stage pipeline overview. Raw multi-patient EEG enters four sequential stages: population-level backbone pretraining (Stage 1), domain-adversarial patient disentanglement via gradient reversal (Stage 2), few-shot MAML personalization to unseen patients (Stage 3), and SHAP, attention, and feature-space interpretability (Stage 4). The dashed line denotes the shared CNN–Transformer backbone, which is refined during Stages 1–3 and queried without updating its weights in Stage 4. The framework returns a clinical-support prediction with an accompanying explanation.

**Alt text:** Flowchart of the REALM cross-patient seizure-detection pipeline. Raw multi-patient EEG passes through CNN–Transformer population pretraining, gradient-reversal patient disentanglement, few-shot MAML adaptation, and SHAP, attention, and feature-space interpretation to produce a clinical prediction and explanation. The backbone is refined during Stages 1–3 and queried without updating it in Stage 4.

**Provenance:** Drawn deterministically by [`scripts/generate_figures.py`](../scripts/generate_figures.py). It encodes the framework specification and has no source dataset or random seed.

## Figure 2. Stage 2 gradient-reversal architecture

**Classification:** Conceptual architecture; no measured data.

**Full caption:** REALM Stage 2 adversarial architecture with a Gradient Reversal Layer (GRL). The shared backbone \(B_\theta\) maps EEG input \(x\) to latent representation \(z\). A seizure-classification head follows the solid path and minimizes \(\mathcal{L}_{\mathrm{seiz}}\). A patient-identity head follows the dashed adversarial path and minimizes positive cross-entropy \(\mathcal{L}_{\mathrm{pat}}\) with respect to the patient-head parameters. The GRL returns \(z\) unchanged during the forward pass and multiplies only the patient-loss gradient propagated to the backbone by \(-\lambda\). The backbone therefore preserves seizure information while adversarially suppressing patient-identifying information. Its objective is \(\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{seiz}}-\lambda\mathcal{L}_{\mathrm{pat}}\); the GRL has no trainable parameters.

**Alt text:** EEG input enters a CNN–Transformer backbone and produces latent vector z. A solid blue branch sends z to a seizure head that minimizes seizure loss. A dashed orange branch sends z through a gradient reversal layer to a patient head that minimizes positive cross-entropy with respect to its own parameters. The reversed gradient causes only the backbone to maximize patient loss. The total backbone objective is seizure loss minus lambda times patient loss.

**Provenance:** Drawn deterministically by [`scripts/generate_figures.py`](../scripts/generate_figures.py). Equations are architectural definitions, not fitted values.

## Figure 3. Simulated feature-space evolution

**Classification:** **SIMULATED — not measured model embeddings.**

**Full caption:** Simulated feature-space geometry evolution during REALM training. Panels 1 and 2 use redundant patient color and marker-shape encodings; open markers denote non-seizure samples and filled markers denote seizure samples. After Stage 1, the space is organized into patient-specific clusters with seizure and non-seizure samples mixed within each cluster. During the adversarial confusion valley, patient boundaries blur. After Stage 2, patient identity is mixed and seizure state becomes the dominant geometry, shown with blue circles for non-seizure and orange stars for seizure. In the final panel, outlined triangles from a simulated unseen patient map into the existing non-seizure and seizure regions. Coordinates are conceptual and must not be interpreted as measured t-SNE, UMAP, or model outputs.

**Alt text:** Four simulated embedding panels show patient-specific clusters after Stage 1, blurred patient boundaries during adversarial training, separate non-seizure and seizure clusters after Stage 2, and samples from a new patient mapping into the two shared seizure-state clusters. These are conceptual coordinates, not measured model embeddings.

**Provenance:** Generated with NumPy seed `31703` by [`scripts/generate_figures.py`](../scripts/generate_figures.py). Each point is preserved in [`figure-03-simulated-embedding.csv`](assets/figure-data/figure-03-simulated-embedding.csv), including panel, coordinates, patient, seizure class, new-patient flag, status, and seed.

## Figure 4. Simulated EEG attribution

**Classification:** **SIMULATED — synthetic EEG and synthetic positive-attribution values.**

**Full caption:** REALM Stage 4 positive-attribution heatmap on a simulated eight-channel EEG trace. The channels Fp1, F3, C3, P3, Fp2, F4, C4, and P4 are shown over a two-second interval. The dashed dark line marks simulated seizure onset at 0.68 seconds and the dotted orange line marks offset at 1.36 seconds. Warm shading denotes higher simulated positive attribution supporting a seizure prediction. Attribution is concentrated over rhythmic ictal activity and weighted most strongly in central and parietal channels, with low attribution outside the ictal interval. EEG signals and attribution values are synthetic and do not represent a patient or measured model output.

**Alt text:** Eight synthetic EEG channels are plotted from zero to two seconds. A seizure begins at 0.68 seconds and ends at 1.36 seconds. Warm orange positive attribution is concentrated over rhythmic ictal activity and is strongest in central and parietal channels. These are simulated EEG and attribution values, not patient data or measured model output.

**Provenance:** Generated with NumPy seed `40404` by [`scripts/generate_figures.py`](../scripts/generate_figures.py). The 3,208 channel-time observations are preserved in [`figure-04-simulated-eeg-shap.csv`](assets/figure-data/figure-04-simulated-eeg-shap.csv) with amplitude units, attribution, seizure state, sampling rate, onset, offset, status, and seed.

## Figure 5. Illustrative few-shot performance

**Classification:** **ILLUSTRATIVE — not empirical performance.**

**Full caption:** Illustrative few-shot adaptation performance versus labelled seizure examples \(K\). The left panel shows hypothetical cross-patient AUC and the right panel shows hypothetical sensitivity for REALM and transfer learning at \(K\in\{1,5,10,20\}\). Solid blue lines with filled circles represent REALM; dashed orange lines with open squares represent transfer learning. Error bars are hypothetical one-standard-deviation values. Neutral reference lines mark hypothetical backbone-only floors and patient-specific ceilings. The values demonstrate the intended evaluation design and narrative pattern only; they are not experimental results and must be replaced with locked evaluation outputs before any performance claim is made. The figure retains the manuscript's positive-example wording; the reference episode sampler instead interprets `support_per_class=K` as \(K\) examples from each class (\(2K\) total), so any empirical replacement must state its support convention explicitly.

**Alt text:** Two illustrative line charts compare REALM with transfer learning at K values of 1, 5, 10, and 20 labelled seizure examples. Hypothetical REALM AUC and sensitivity remain above transfer learning and approach hypothetical patient-specific references as K increases. Error bars are hypothetical one-standard-deviation values; no empirical performance is shown.

**Provenance:** Fixed hypothetical values are stored in [`figure-05-illustrative-performance.csv`](assets/figure-data/figure-05-illustrative-performance.csv) and rendered by [`scripts/generate_figures.py`](../scripts/generate_figures.py). The CSV labels every estimate and reference as `ILLUSTRATIVE`; no random seed is used.

## Reuse requirements

- Keep each full caption adjacent to its figure when possible.
- Use the supplied alt text or a context-specific equivalent; do not use the filename as alt text.
- Preserve the visible `SIMULATED` and `ILLUSTRATIVE` badges while the linked source rows retain those statuses.
- Do not cite Figures 3–5 as empirical evidence. Figure 5 in particular is an evaluation mock-up until its CSV is replaced by reviewed experiment outputs.
- When adapting colors, preserve the redundant marker, fill, line-style, direct-label, and ordering channels defined in the contracts.
