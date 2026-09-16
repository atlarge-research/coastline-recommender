# Changelog

All notable changes to Coastline are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Coastline adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version lives only in
`pyproject.toml`; release tags mirror it as `vX.Y.Z`.

## [Unreleased]

### Added

- Test matrix now covers **macOS** as well as Linux (`ubuntu-latest`, `macos-latest`, Python 3.13),
  with `fail-fast: false` so one platform's failure cannot hide the other's. Both runners sync
  `--all-extras`; the informational `-m ml_isolated` leg stays Linux-only. `wheel-smoke`,
  `docker-build` and `docs` stay Linux-only — they exercise the shipped Linux artifacts.
- **Coverage reporting**: coverage settings in `pyproject.toml` (`uv run pytest --cov`), and a
  per-leg upload to Codecov via `codecov/codecov-action@v5` over OIDC — no upload token required.
  The upload is not a gate (`fail_ci_if_error: false`). `pytest-cov` was already in the dev group
  but never invoked.
- **`.pre-commit-config.yaml`** running ruff (`--fix`) and `ruff-format` pinned to the same
  `v0.15.15` the dev group pins, plus `end-of-file-fixer` and `trailing-whitespace`. CI runs
  `pre-commit run --all-files`, so the hooks and CI cannot drift.
- **`.github/dependabot.yml`**: weekly, grouped updates for the `github-actions` and `uv`
  ecosystems, with runtime and development dependency bumps split into separate PRs.
- **`CITATION.cff`** (validated against CFF schema 1.2.0) with a `preferred-citation` for the MSc
  thesis, and **`.zenodo.json`** so an archived release is attributed correctly. The Zenodo file
  takes effect only once the GitHub-Zenodo integration is enabled for the repository.
- **PEP 740 attestations** on the PyPI publish step: the sdist and wheel are signed with the
  workflow's Sigstore identity, so PyPI can attest the artifact came from this repo at this tag.
- README badges (CI, Codecov, PyPI, licence) and this changelog.

### Changed

- The release workflow now triggers on **plain semver tags only** (`v[0-9]+.[0-9]+.[0-9]+`). The
  previous `v*.*.*` glob also matched suffixed marker tags, so a freeze tag such as `v0.2.2-thesis`
  would have started a real publish run.

### Fixed

- Dropped the `Programming Language :: Python :: 3.14` classifier: `requires-python` caps the
  distribution at `<3.14`, so the classifier advertised an interpreter pip refuses to install on.
- Trailing whitespace and missing final newlines across `docs/`, `mkdocs.yml` and the UI's
  `architecture.svg` (found by the new whitespace hooks).

## [0.2.2] - 2026-09-16

The thesis freeze: the state of Coastline used for the MSc thesis experiments, tagged
`v0.2.2-thesis` — a marker tag, deliberately outside the release trigger, so archiving the freeze
never publishes it.

### Added

- **Parallel pipeline stages**: the recommendation stages run as fork-join phases.
- **Opt-in empirical OOM guard** on top of the AutoConf feasibility gate.
- Centralised enums and defaults in `constants.py`; the alias vocabulary is gone.
- The sklearn portfolio collapsed into one generic predictor (and one generic trainer), plus the
  `CODE_QUALITY.md` standard (10 functional + 5 non-functional requirements).

### Fixed

- **Per-device batch semantics across the AutoConf boundary**: `WorkloadSpec.batch_size` is
  per-device while AutoConf's `JobConfig.batch_size` is the total effective batch; the conversion
  now happens explicitly at the boundary, where it previously changed every feasibility verdict
  silently.

### Changed

- The freeze tag pins **Kavier** to its own thesis tag `v0.5.1-thesis` (a git source) instead of the
  PyPI wheel `0.5.0.2`, so the reproducibility capsule replicates against the code the published
  results were produced with. `main` keeps the `kavier>=0.5,<0.6` PyPI range.

[Unreleased]: https://github.com/atlarge-research/coastline-recommender/compare/v0.2.2-thesis...HEAD
[0.2.2]: https://github.com/atlarge-research/coastline-recommender/releases/tag/v0.2.2-thesis
