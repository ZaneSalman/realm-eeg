# Data governance and clinical safety

This repository contains software and synthetic examples only. It does not authorize access
to health records, waive a data-use agreement, or establish that de-identified EEG can be
published.

## Before using real EEG

- Confirm institutional review, consent, data-use agreement, and local privacy requirements.
- Store raw EEG and linkage keys in approved encrypted systems with least-privilege access.
- Keep patient identifiers, exact dates, free-text reports, file headers, and provenance that
  can re-identify a person out of public logs and version control.
- Review EDF headers and sidecars for residual protected information before any transfer.
- Record dataset version, hashes, transformations, label provenance, and authorized users.
- Define retention, deletion, incident-response, and model-checkpoint handling policies.

Model checkpoints and latent embeddings may retain information about training patients. Treat
them as derived sensitive artifacts until a documented privacy assessment shows otherwise.
Patient-adapted checkpoints should never be published by default.

## Evaluation gates

Do not claim clinical validity from the synthetic demo or the illustrative repository figures.
A scientific evaluation should include, at minimum:

1. locked patient-independent test sets and an auditable split manifest;
2. external-site validation with acquisition and population shift analysis;
3. sensitivity, specificity, precision, F1, AUC, calibration, and false-alarm events per hour;
4. per-patient estimates with confidence intervals rather than pooled windows alone;
5. missing-channel, artifact, seizure-type, age, sex, race/ethnicity, and site subgroup audits
   where lawful and statistically supportable;
6. comparison against prespecified population, transfer-learning, adversarial, and
   patient-specific baselines;
7. patient-identity leakage probes on frozen embeddings; and
8. clinician review of both correct and failed predictions, without treating attention or
   attribution as causal evidence.

## Intended boundary

REALM is a proposed decision-support research framework. It is not autonomous diagnosis,
continuous monitoring software, a certified medical device, or a replacement for expert EEG
interpretation. Any move toward prospective clinical use requires a separate quality system,
human-factors work, cybersecurity assessment, monitoring plan, regulatory analysis, and
validation in the intended setting.

