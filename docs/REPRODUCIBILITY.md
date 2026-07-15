# Reproducibility protocol

The manuscript describes the REALM objectives but does not specify enough engineering detail
to reproduce an empirical result. This repository therefore separates **manuscript-defined
semantics** from **reference defaults** selected for executable code.

## Deterministic smoke test

```bash
python -m pip install -e '.[dev,figures]'
pytest
realm-eeg synthetic-demo --seed 17 --output realm-demo
python scripts/generate_figures.py
```

The demo is deliberately small and synthetic. Repeating it checks software execution, not
model quality or bitwise determinism across hardware and library versions.

The reference demo starts a fresh AdamW optimizer at each stage boundary and writes model-only
checkpoints after Stages 1, 2, and 3. Optimizer moments are intentionally not carried across
stages or serialized, so resuming a stage from its predecessor checkpoint reproduces this
boundary policy.

## Run record for empirical work

Archive the following for every run:

- git commit and uncommitted diff;
- Python, PyTorch, CUDA/MPS, driver, and dependency versions;
- hardware and precision mode;
- random seeds and deterministic-kernel settings;
- exact configuration and preprocessing implementation;
- immutable dataset version and source-file hashes;
- patient-level train/meta-validation/test manifest;
- support/query samples for every task;
- checkpoint-selection rule and all evaluated thresholds;
- window-to-event post-processing; and
- per-patient predictions sufficient to recompute every reported metric.

Do not select thresholds or early-stopping checkpoints on the final test patients. If Stage 3
uses labels from a test patient, designate those examples as support before seeing query
outcomes and report `K` unambiguously.

## Reference choices requiring sensitivity analysis

The executable defaults include a 4-second, 256 Hz, 19-channel input; a three-block CNN; a
two-layer Transformer with deterministic sinusoidal token positions; logistic GRL scheduling;
positive patient cross-entropy through the GRL; first-order MAML; adaptation of the backbone
and seizure head; zero-baseline integrated gradients; and Euclidean centroid distance. None of
those exact values or variants is fully specified by the manuscript.

At minimum, vary window length/overlap, channel montage, filter policy, class weighting,
latent width, adversarial schedule/weight, which layers adapt, support composition, inner
steps/rate, attribution baseline, alert threshold, and event-merging rules.

## Reporting

Report the number of people, recordings, hours, seizures, and windows in every partition.
Include exclusions and missingness. Provide patient-level bootstrap confidence intervals and
paired comparisons when assumptions are met. The manuscript calls false alarms per hour
“FDR”; code and reports should use `false_alarm_events_per_hour` to avoid confusion with the
conventional false discovery rate.

Compute that event metric only from a continuous timeline. The implementation requires an
explicit recording ID and window start time for every score, a prespecified gap-merging rule,
and the actual total recording hours. It intentionally cannot infer recording duration from a
balanced, shuffled, or overlapping evaluation array. The synthetic demo therefore leaves the
metric unset.
