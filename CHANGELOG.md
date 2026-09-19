# Changelog

All notable changes to Coastline are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Coastline adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version lives only in
`pyproject.toml`; release tags mirror it as `vX.Y.Z`.

## [Unreleased]

## [0.2.3] - 2026-09-19

### Added

- Test matrix now covers **macOS** as well as Linux (`ubuntu-latest`, `macos-latest` × Python 3.11, 3.12, 3.13),
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
- **`Coastline(empirical_oom_guard=True)`**: the public facade can now opt into the empirical
  per-device token-budget gate (`EMPIRICAL_OOM_TOKEN_BUDGET`, 60,224 tokens/device, fitted to the
  observed OOMs), layered on top of whichever feasibility backend is selected. Until now the gate
  was reachable only through the config-driven `predictors.empirical_oom_guard` key, so the
  `rules` backend — structural sanity guards only, no memory model — had no OOM protection at all
  when driven through `import coastline`. Off by default; it only ever turns feasible into
  infeasible.

### Changed

- The release workflow now triggers on **plain semver tags only** (`v[0-9]+.[0-9]+.[0-9]+`). The
  previous `v*.*.*` glob also matched suffixed marker tags, so a freeze tag such as `v0.2.2-thesis`
  would have started a real publish run.
- **`requires-python` lowered to `>=3.11,<3.14`** (was `>=3.13,<3.14`), with `ruff`
  `target-version = "py311"` and `mypy` `python_version = "3.11"`, plus 3.11/3.12 classifiers.
  No source change was needed — all 88 modules already parsed under the 3.11 grammar and use no
  3.12+ stdlib — and the full test suite passes on 3.11. This is what lets COASTLINE be consumed
  by IBM's `ado`, whose virtualenv is Python 3.11. The `<3.14` cap is unchanged and still comes
  from `ado-autoconf` -> `autogluon` 1.5, which caps `pyarrow<21`.
- Kavier requirement tightened to **`kavier>=0.5.2,<0.6`**. The previous `>=0.5,<0.6` resolved to
  the PyPI wheel 0.5.0.2 for anyone installing from the index — which predates the
  `distributed-*` / `consolidated-*` policy rename and the MFU/goodput additions — while local
  development silently got something newer through a `[tool.uv.sources]` git pin that is absent
  from published metadata. Both now agree on the same published release.

### Removed

- The `[tool.uv] override-dependencies = ["pyarrow>=23.0.1"]` escape hatch, and the
  `[tool.uv.sources]` git pin of Kavier. The override's premise was stale: Kavier relaxed its own
  floor to `pyarrow>=15` precisely so `autogluon`'s `<21` cap could be satisfied, so forcing
  `>=23` only produced a resolution `autogluon` does not support. Neither setting propagated into
  the published wheel, so removing them makes a local `uv sync` resolve what a consumer actually
  gets (`pyarrow==20`).
- **`dev/ado_plugin/`**. The ado experiment plugin now lives in IBM's `ado` repository, under
  `plugins/custom_experiments/coastline`, where it is packaged and tested against ado core
  directly. The copy kept here had diverged from it (different packaging, a gutted bridge, a
  different test suite) and was the stale one. Removed with it: the `dev/ado_plugin` entry in
  pytest's `pythonpath`, and the CI step that ran the plugin's suite in the root environment.

### Fixed

- Dropped the `Programming Language :: Python :: 3.14` classifier: `requires-python` caps the
  distribution at `<3.14`, so the classifier advertised an interpreter pip refuses to install on.
- Trailing whitespace and missing final newlines across `docs/`, `mkdocs.yml` and the UI's
  `architecture.svg` (found by the new whitespace hooks).
- The stale "divisibility" description of the `rules` feasibility backend, everywhere it
  survived: `CLAUDE.md`, the refusal message raised when `feasibility=autoconf` is asked for
  without AutoConf, the `--feasibility` help of `simulate` / `explain` / `recommend-trace`, the
  reason string `recommend-trace` writes into trace output, the `FeasibilityMode.RULES` comment,
  the sample configs, the docs pages and the test docstrings. The
  `batch_size % total_gpus` rule was deliberately dropped when `batch_size` became per-device — a
  per-device batch has nothing left to divide — so `RulesFeasibilityChecker` is structural sanity
  guards only (`total_gpus >= 1`, `batch_size >= 1`): no memory model, no OOM check. Every one of them
  now says that, and point at `empirical_oom_guard` for the per-device token budget that can be
  layered over it.

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

[Unreleased]: https://github.com/atlarge-research/coastline-recommender/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/atlarge-research/coastline-recommender/compare/v0.2.2-thesis...v0.2.3
[0.2.2]: https://github.com/atlarge-research/coastline-recommender/releases/tag/v0.2.2-thesis
