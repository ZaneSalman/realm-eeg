# Public Release Checklist

Do not make the repository public or publish a release until every blocking item
below is complete and documented. A checked box must represent an affirmative,
recorded decision rather than an assumption.

## 1. Authorship, review, and intellectual property

- [ ] Every manuscript/framework coauthor has approved the repository contents,
      authorship order, citation metadata, licenses, and public-release timing.
- [ ] The target journal/editor has confirmed that a public author-branded
      repository will not compromise double-anonymous review.
- [ ] Any applicable preprint, code-sharing, and embargo rules have been checked
      for the active submission venue.
- [ ] Each contributor has confirmed authority to license their contribution.
- [ ] Relevant institutional intellectual-property or technology-transfer
      offices have approved public disclosure, including any patent review.
- [ ] The decision and date for each approval above are retained outside the
      public repository.

## 2. Repository identity and citation

- [ ] The final public URL has been added as `repository-code` in `CITATION.cff`
      and under `[project.urls]` in `pyproject.toml`.
- [ ] `rg -n "example[.]com|INSERT CONTACT|TODO" . --glob '!RELEASE_CHECKLIST.md'`
      returns no unresolved public-release placeholders.
- [ ] `CITATION.cff` validates against Citation File Format 1.2.0.
- [ ] Version `0.1.0` agrees across package metadata, citation metadata,
      changelog, documentation, tag, and release notes; the actual public release
      date has been added to `CITATION.cff` and the changelog.
- [ ] The manuscript is described as unpublished; no article DOI, journal
      acceptance, or publication status has been invented.
- [ ] The manuscript's code-availability statement contains the final public URL
      only after the repository is actually public.

## 3. Blinding, privacy, and repository hygiene

- [ ] No raw manuscript drafts, blinded submission files, title pages, reviewer
      correspondence, email attachments, or working PDFs are present.
- [ ] No private address, phone number, personal email, credentials, tokens,
      signed download URLs, account identifiers, or other personal metadata are
      present in files or Git history.
- [ ] No `.DS_Store`, `._*`, `__MACOSX`, editor swap files, or extended-attribute
      sidecar files are present.
- [ ] Public figures and documents were cleanly exported; macOS extended
      attributes and download-origin metadata were removed from release copies.
- [ ] Git history, tags, issues, and release artifacts have been scanned for
      secrets, personal information, private correspondence, and oversized
      binaries.
- [ ] A clean clone contains only files intentionally approved for public use.

## 4. Data, models, and scientific claims

- [ ] No raw EEG, clinical data, annotations, dataset-derived samples, local data
      indexes, or controlled-access metadata are committed.
- [ ] No trained weights, checkpoints, optimizer states, experiment caches, or
      artifacts derived from restricted data are committed.
- [ ] Examples and automated tests run using synthetic or explicitly
      redistributable fixtures only.
- [ ] Dataset documentation pins exact versions and gives upstream citations,
      licenses, access terms, and user-supplied path instructions.
- [ ] TUSZ instructions require users to obtain access and accept its data-use
      agreement themselves; the repository does not redistribute it.
- [ ] Any future weights have completed a separate data-license, institutional,
      privacy, and model-risk review before publication.
- [ ] Figures based on simulated or illustrative values say
      `SIMULATED/ILLUSTRATIVE — NOT EMPIRICAL RESULTS` in the figure and caption.
- [ ] All pre-REALM labels have been replaced with approved `REALM` branding.
- [ ] README, model card, examples, and release notes make no unsupported
      accuracy, sensitivity, clinical-performance, or regulatory claims.
- [ ] Research-only and not-for-clinical-use limitations are prominent.

## 5. Licensing and provenance

- [ ] Software files are covered by Apache-2.0 and the root `LICENSE` is present.
- [ ] Original documentation, figures, and tables are marked CC BY 4.0 and
      `LICENSES/CC-BY-4.0.txt` is present.
- [ ] The manuscript is not represented as covered by the software license.
- [ ] Third-party assets, dependencies, and copied code have documented source,
      version, attribution, and license compatibility.
- [ ] Dataset licenses are documented but no upstream dataset is relicensed.
- [ ] Figure-generation sources, data provenance, and random seeds are included.
- [ ] AI-assisted figure or text generation is disclosed where required by the
      target journal, institutions, or contributor agreements.

## 6. Quality and reproducibility

- [ ] Installation succeeds in a clean supported environment.
- [ ] Tests, linting, and formatting checks pass from a clean clone; any type
      checker adopted by the project also passes.
- [ ] The synthetic quickstart completes without network credentials or private
      data and reproduces its documented outputs.
- [ ] Patient-level splitting and leakage controls have automated tests.
- [ ] Random seeds, configuration files, dependency bounds, and environment
      information needed for reproducibility are recorded.
- [ ] Security reporting is enabled privately on the repository host.
- [ ] Documentation links and figure/table references resolve.

## 7. Final publication gate

- [ ] A release candidate has been reviewed by at least one person other than the
      person who assembled it.
- [ ] A source archive made from a clean clone contains no ignored or untracked
      local material.
- [ ] Release notes clearly identify version `0.1.0` as an unpublished research
      prototype without completed patient-level validation.
- [ ] The release tag is signed or otherwise verified using the repository's
      approved release process.
- [ ] Optional archival DOI integration is configured only after the final
      repository URL and authorship metadata are approved.
- [ ] All blocking approvals in Section 1 remain valid on the publication date.
