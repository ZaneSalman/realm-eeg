# REALM Figure Contracts

This document defines the analytical, provenance, accessibility, and export contract for every publication figure in the REALM repository. The generated files are documentation assets, not evidence beyond the status stated on each figure.

## Regeneration

From the repository root:

```bash
python scripts/generate_figures.py
```

The plotting code requires NumPy and Matplotlib. Install the project’s `figures` optional
dependency group. To deterministically rebuild the simulated and illustrative CSV files as
well as the figures:

```bash
python scripts/generate_figures.py --refresh-data
```

Without `--refresh-data`, the generator reads the committed CSVs so that reviewed rows remain the source of record.

## Shared visual and export contract

- Surface: static documentation and manuscript figures.
- Renderer: reproducible Matplotlib script using the non-interactive `Agg` backend.
- Exports: 300 dpi PNG and text-preserving SVG in `docs/assets/figures/`.
- Background: white or near-white, with quiet neutral guides and dark ink typography.
- Palette roots: blue `#2F6B9A`, orange `#D97706`, gold `#B6860B`, olive `#667A35`, and pink `#A34F73`, plus neutral ink and gray.
- Accessibility: no state or series is distinguished by color alone. Shape, fill, line style, direct text, ordering, or faceting supplies a redundant channel.
- SVG accessibility: every SVG receives a `<title>`, `<desc>`, `role="img"`, and `aria-labelledby` relationship.
- Naming: file stems identify figure number, subject, and provenance status where applicable.
- Provenance labels: `SIMULATED` and `ILLUSTRATIVE` are printed within the visual, included in the source rows, and repeated in this contract.
- Quality gate: titles, subtitles, legends, annotations, reference lines, and status badges must remain legible in both PNG and SVG exports; no clipping, collisions, detached labels, or color-only distinctions are acceptable.

## Figure 1 — REALM pipeline

- Outputs: `figure-01-realm-pipeline.png` and `figure-01-realm-pipeline.svg`.
- Status: **CONCEPTUAL ARCHITECTURE**. No measured data are shown.
- Analytical question: How does information and the shared backbone move through REALM?
- Takeaway: population pretraining, patient-identity disentanglement, and few-shot adaptation refine one carried-forward backbone during Stages 1–3; Stage 4 queries that backbone without updating it.
- Family and variant: process flow / staged architecture diagram.
- Data sufficiency: not applicable; the visual is an architectural specification.
- Encoding: ordered boxes and arrows establish sequence; stage number, heading, and explanatory text identify every stage independently of color. A dashed line with repeated nodes identifies the shared backbone and explicitly distinguishes Stage 4 querying from Stages 1–3 refinement.
- Source-data need: none. The diagram itself is fully defined in `scripts/generate_figures.py`.
- Alt text: “Flowchart of the REALM cross-patient seizure-detection pipeline. Raw multi-patient EEG passes through CNN–Transformer population pretraining, gradient-reversal patient disentanglement, few-shot MAML adaptation, and SHAP, attention, and feature-space interpretation to produce a clinical prediction and explanation. The backbone is refined during Stages 1–3 and queried without updating it in Stage 4.”

## Figure 2 — Gradient-reversal architecture

- Outputs: `figure-02-gradient-reversal-architecture.png` and `figure-02-gradient-reversal-architecture.svg`.
- Status: **CONCEPTUAL ARCHITECTURE**. No measured data are shown.
- Analytical question: How does Stage 2 preserve seizure information while suppressing patient identity?
- Takeaway: both classification heads minimize positive cross-entropy with respect to their own parameters, while the GRL reverses only the patient-loss gradient reaching the backbone, causing the backbone to adversarially maximize patient-classification loss.
- Family and variant: branched architecture diagram with objective callout.
- Data sufficiency: not applicable; the visual is an architectural specification.
- Encoding: the seizure branch is a solid blue path with circular marker; the adversarial branch is a dashed orange path with diamond marker. The patient-loss box says `HEAD MIN`, while the separate backbone objective and explanatory text show adversarial maximization through the reversed gradient. Branch labels, formulas, and direct objective labels remain interpretable without color.
- Source-data need: none. Equations are native plot text and are defined in `scripts/generate_figures.py`.
- Alt text: “EEG input enters a CNN–Transformer backbone and produces latent vector z. A solid blue branch sends z to a seizure head that minimizes seizure loss. A dashed orange branch sends z through a gradient reversal layer to a patient head that minimizes positive cross-entropy with respect to its own parameters. The reversed gradient causes only the backbone to maximize patient loss. The total backbone objective is seizure loss minus lambda times patient loss.”

## Figure 3 — Feature-space evolution

- Outputs: `figure-03-feature-space-evolution-simulated.png` and `figure-03-feature-space-evolution-simulated.svg`.
- Status: **SIMULATED — NOT MEASURED EMBEDDINGS**.
- Analytical question: What representational geometry is REALM designed to create across Stages 1–3?
- Takeaway: patient-organized clusters blur during adversarial training and are replaced by seizure-state geometry into which an unseen patient can map.
- Family and variant: four-panel scatter small multiple.
- Data sufficiency: 42 points per patient in each of the first two panels, 240 points in the Stage 2 panel, and 212 points in the new-patient panel. This density is sufficient to show conceptual cluster geometry, but it is not empirical evidence.
- Data source: `docs/assets/figure-data/figure-03-simulated-embedding.csv`.
- Grain: one simulated embedding point per row.
- Fields: panel, point ID, x/y coordinate, patient ID, seizure class, new-patient flag, provenance status, and random seed.
- Encoding: in panels 1–2, patient identity uses both color and marker shape; non-seizure is open and seizure is filled. In panels 3–4, class uses both color and marker shape, while new-patient samples use triangles with dark outlines.
- Fallback: if measured embeddings become available, replace the CSV with held-out model output at the same row grain and remove the simulated badge only after validating provenance.
- Alt text: “Four simulated embedding panels show patient-specific clusters after Stage 1, blurred patient boundaries during adversarial training, separate non-seizure and seizure clusters after Stage 2, and samples from a new patient mapping into the two shared seizure-state clusters. These are conceptual coordinates, not measured model embeddings.”

## Figure 4 — EEG attribution

- Outputs: `figure-04-shap-eeg-simulated.png` and `figure-04-shap-eeg-simulated.svg`.
- Status: **SIMULATED — SYNTHETIC EEG AND ATTRIBUTION**.
- Analytical question: What would a clinically legible positive-attribution overlay look like during an ictal interval?
- Takeaway: simulated seizure-supporting attribution is concentrated over rhythmic ictal activity and remains low outside the marked interval.
- Family and variant: aligned eight-channel signal traces with channel-level heatmap backgrounds.
- Data sufficiency: 401 time points per channel across eight channels at 200 Hz, yielding 3,208 source rows. This is sufficient for a smooth two-second conceptual trace.
- Data source: `docs/assets/figure-data/figure-04-simulated-eeg-shap.csv`.
- Grain: one channel-time observation per row.
- Fields: time, channel, signal amplitude in microvolts, positive attribution, seizure state, sampling rate, onset, offset, provenance status, and random seed.
- Encoding: waveform shape provides the physiological signal; orange intensity encodes positive attribution; onset and offset use different line styles plus direct labels. A scale bar and colorbar provide units.
- Fallback: replace the CSV with appropriately de-identified, consented empirical output before describing the visual as patient data or model performance.
- Alt text: “Eight synthetic EEG channels are plotted from zero to two seconds. A seizure begins at 0.68 seconds and ends at 1.36 seconds. Warm orange positive attribution is concentrated over rhythmic ictal activity and is strongest in central and parietal channels. These are simulated EEG and attribution values, not patient data or measured model output.”

## Figure 5 — Few-shot performance

- Outputs: `figure-05-few-shot-performance-illustrative.png` and `figure-05-few-shot-performance-illustrative.svg`.
- Status: **ILLUSTRATIVE — NOT EMPIRICAL PERFORMANCE**.
- Analytical question: What experimental comparison should a few-shot evaluation communicate once measured results exist?
- Takeaway: the illustrative pattern shows REALM improving with K and retaining its largest advantage over transfer learning in the lowest-label regime.
- Family and variant: two-panel uncertainty-and-benchmark line chart.
- Data sufficiency: four ordered K values per method and metric. This is an intentionally sparse discrete dose-response comparison, not a continuous temporal trend.
- Data source: `docs/assets/figure-data/figure-05-illustrative-performance.csv`.
- Grain: one hypothetical estimate or one hypothetical reference per row.
- Fields: record type, metric, K, method, mean, standard deviation, reference name/value, unit, status, and explanatory note.
- Encoding: REALM uses a solid blue line with filled circles; transfer learning uses a dashed orange line with open squares. Reference lines use neutral dashed/dotted strokes and direct labels. The subtitle states the focused 0.50–1.00 scale.
- Support convention: the manuscript-style figure treats \(K\) as labelled seizure examples, while the reference sampler uses \(K\) examples per class (\(2K\) total). Any empirical replacement must label the convention explicitly.
- Fallback: replace all estimate, uncertainty, and reference rows with outputs from the locked evaluation pipeline before making any performance claim. Do not remove the illustrative badge while hypothetical rows remain.
- Alt text: “Two illustrative line charts compare REALM with transfer learning at K values of 1, 5, 10, and 20 labelled seizure examples. Hypothetical REALM AUC and sensitivity remain above transfer learning and approach hypothetical patient-specific references as K increases. Error bars are hypothetical one-standard-deviation values; no empirical performance is shown.”

## Review checklist

Before publishing regenerated files:

1. Confirm every title says REALM and contains no pre-REALM branding.
2. Confirm Figure 3 and Figure 4 visibly say `SIMULATED` and Figure 5 visibly says `ILLUSTRATIVE`.
3. Confirm the CSV status fields match the visual badges.
4. Review PNG exports at 100% and 200% for clipping, overlap, low contrast, and tiny type.
5. Open SVG exports and confirm text is selectable and the root element contains accessible title/description metadata.
6. Check all distinctions in grayscale: box labels and stage numbers in Figure 1; solid/circle versus dashed/diamond in Figure 2; marker/fill distinctions in Figure 3; waveform/line-style cues in Figure 4; and line/marker/reference styles in Figure 5.
7. If empirical data replace simulated or illustrative rows, update the status badge, contract, caption, alt text, and source provenance together.
