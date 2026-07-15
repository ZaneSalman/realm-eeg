# Contributing to REALM EEG

Thank you for helping improve REALM EEG. This repository is an unpublished
research reference implementation. It is not a validated clinical system or a
medical device.

## Before contributing

- Search existing issues and pull requests before opening a new one.
- For substantial changes, open an issue describing the proposed design and
  validation plan before implementation.
- Do not upload patient data, EEG recordings, annotations, credentials, model
  checkpoints, private manuscripts, reviewer correspondence, or other
  controlled material.
- Use synthetic fixtures in tests. A contribution that requires restricted data
  must describe how maintainers can test it without receiving that data.
- Do not make clinical-performance claims without reproducible, patient-level
  evidence and appropriate review.

## Development workflow

1. Create a virtual environment and install the project with its development
   dependencies.
2. Create a focused branch for one change.
3. Add or update tests and documentation with the implementation.
4. Run the configured tests, formatter check, and linter; run any type checker
   configured for the contribution.
5. Open a pull request that explains the problem, approach, risks, and
   verification performed.

Keep patient-level splitting and leakage prevention explicit in all data and
evaluation code. Random segment-level splits are not an acceptable substitute
for patient-independent evaluation.

## Pull-request checklist

- [ ] The change is scoped and documented.
- [ ] Tests cover expected behavior and important failure cases.
- [ ] Tests use synthetic or redistributable data only.
- [ ] No secrets, personal information, raw EEG, checkpoints, or private drafts
      are included.
- [ ] New dependencies are necessary, version-bounded, and license-compatible.
- [ ] Public APIs and configuration changes are documented.
- [ ] Simulated or illustrative outputs are clearly labeled as such.
- [ ] The change does not imply clinical validation, regulatory clearance, or
      fitness for diagnosis.

## Licensing contributions

Software contributions are submitted under the Apache License 2.0. Original
documentation, figures, tables, and figure source-data CSVs are submitted under Creative Commons
Attribution 4.0 International unless a file states otherwise. Contributors must
have authority to submit their work and must identify third-party material and
its license in the pull request.

By contributing, you represent that the contribution is your original work or
that you have sufficient rights to submit it under the applicable repository
license. Maintainers may request a Developer Certificate of Origin sign-off:

```text
Signed-off-by: Your Name <an address you are authorized to use>
```

Do not add another person's identity or sign-off without authorization.

## Conduct and security

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report
security issues using the private process in [SECURITY.md](SECURITY.md), not a
public issue containing exploit details or sensitive information.
