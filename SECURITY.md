# Security Policy

## Scope

REALM EEG is research software. It has not been clinically validated, cleared,
or approved for diagnosis, treatment, monitoring, or other patient-care use.
Security reports are nevertheless important, especially when a flaw could
cause data leakage, unsafe model loading, arbitrary code execution, dependency
compromise, or misleading evaluation results.

## Supported versions

Security fixes are provided for the latest published release line only. During
the initial release cycle, that is version `0.1.x`.

## Reporting a vulnerability

Use the repository host's private vulnerability-reporting or security-advisory
workflow. Include:

- affected version and component;
- a concise impact statement;
- reproduction steps or a minimal synthetic proof of concept;
- suggested remediation, if known; and
- whether any data or credentials may have been exposed.

Do not include patient data, private EEG data, access tokens, credentials, or
other sensitive material in a report. If private reporting is not enabled, open
a minimal public issue asking the maintainers to establish a private channel;
do not disclose vulnerability details in that issue.

Maintainers will acknowledge a complete report, investigate it, coordinate a
fix and disclosure timeline, and credit the reporter if requested and safe.
Timelines depend on severity and reproducibility.

## Out of scope

- claims based only on use in a clinical environment, because clinical use is
  unsupported;
- vulnerabilities in upstream services with no demonstrated impact on this
  project;
- reports that require disclosure of controlled health data; and
- model-performance disagreements without a reproducible software or data
  integrity defect.

## Safe use

Never load untrusted serialized model files. Keep dataset credentials outside
the repository, minimize access to controlled datasets, pin dependencies, and
review generated artifacts before sharing them.

