# Changelog

All notable changes to Coastline are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Coastline adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version is set in
`pyproject.toml`, `CITATION.cff` and `.zenodo.json`, and release tags match it as `vX.Y.Z`.

## [Unreleased]

### Added

- The Zenodo DOI (10.5281/zenodo.23119267, all versions) in the README, `CITATION.cff` and the docs.
- One table of other goal and preset names for every entry point (`coastline.recommend`, `coastline(...).recommend`,
  `recommend_csv`, `recommend-job`, `recommend-trace`, `explain` and the dashboard): `energy-saver`, the thesis
  name of `energy`, `runtime` and `throughput` for `performance`, and `min-gpu` and `fewest` for `min_gpu`. Letter
  case does not matter, and `-`, `_` and a space between words read alike.
- `coastline.recommend` reads a row's `gpus_per_node` and `number_of_nodes` as the job's own layout under `min_gpu`;
  the other goals ignore them.
- `NoFeasibleGPUCountError`, a `RuntimeError` raised when `min_gpu` finds no feasible GPU count; the dashboard
  says so.

### Changed

- The default goal is `performance` on every entry point: `coastline.recommend`, `coastline(...).recommend`,
  `explain`, `recommend-trace`, `recommend-job`, `recommend_csv`, the bundled configs and the dashboard. A config
  without a preset ranks with it, and the rationale names it.
- `alpha` is the performance (runtime) weight and `beta` the energy weight, as in the thesis: performance
  0.8/0.2, balanced 0.5/0.5, energy 0.2/0.8. The presets rank the same configurations first; a caller's own
  `alpha` and `beta` swap meaning.
- `min_gpu` follows IBM AutoConf's min-GPU recommender, as in the thesis. It keeps the job's total batch
  (per-device batch x `gpus_per_node` x `number_of_nodes`, 1 GPU without a layout) and returns the first feasible
  GPU count in 1, 2, 4, ... with that batch split evenly. It simulates only its picks, drops a pick it cannot
  predict, and ignores the grid and `max_slowdown`.
- `min_gpu` returns one configuration unless `top_k` is set. `experiment.yaml` and the packaged
  `default_experiment.yaml` leave `top_k` unset, so multi-objective returns 5 with them.
- `recommend-trace` reads `metadata.batch_size` as each job's total batch for every goal and writes back the
  recommended per-device batch x GPUs. Without a per-device batch column, the weighted goals start from the total
  over the job's GPUs, and a row whose total does not split evenly is kept unchanged with a note. `--goal min_gpu`
  keeps each job's total batch, and an unknown goal fails before any row runs. The totals in `sample_trace.csv`
  split evenly over their GPUs.
- `explain` renames the score columns `t_score` to `r_score` (runtime) and `p_score` to `e_score` (energy) and
  prints them in the order of the weights. Its `min_gpu` table has no score columns.
- `GridWorkflowPipeline` checks its selection policy when it is built, in any letter case, and raises `ValueError`
  listing the valid ones. `rank_candidates` raises `ValueError` for a policy other than energy, balanced or
  performance.
- `coastline.recommend` and the REPL fail a `max_gpus` below 1 with `max_gpus must be >= 1`, as
  `coastline(...).recommend` does.
- In `coastline.recommend`, an integer column takes a whole number written as a float, such as `8.0`. A value below
  1 or a fraction fails the row with a message that names the column.

## [0.2.3] - 2026-10-03

### Added

- Citation metadata with the three authors in `CITATION.cff` and `.zenodo.json`; Zenodo archives each GitHub
  release.
- Documentation site (`mkdocs.yml`, `docs/`): usage, and how a recommendation is made. CI builds it on every pull
  request; `docs-deploy.yml` publishes it to GitHub Pages.
- `LICENSE-TabPFN.txt`, and "Built with PriorLabs-TabPFN" in the README, the docs and the dashboard, as the
  TabPFN licence requires.
- `Coastline(empirical_oom_guard=True)`: the per-device token budget (60,224 tokens per device) on top of any
  feasibility backend. Off by default.
- `coastline.recommend` takes `max_gpus_per_node` and returns a `recommended_batch_size` column; `recommend_csv`
  writes an `error` column with the reason for every failed row.
- `NoPredictionError`, raised when every configuration that passes the feasibility check gets no usable
  prediction. It names the predictor that failed and its reason.
- A warning when AutoConf judges a GPU outside its training data (it was trained on L40S, NVIDIA-A100-80GB-PCIe,
  NVIDIA-A100-SXM4-80GB and NVIDIA-H100-PCIe), since its OOM verdict is then an extrapolation.

### Changed

- Requires Kavier 0.5.3 from PyPI (`kavier>=0.5.3,<0.6`) instead of a git pin, and Python `>=3.11,<3.14`.
- Predictor, preset and fallback names are case-insensitive, and an unknown name raises `ValueError` instead of
  falling back to a default.
- Model names in Kavier's catalog resolve in any letter case and from HuggingFace ids, `fine_tuning_method` is
  lowercased, and `feasibility_model` is canonicalised like `llm_model`.
- GPU names follow Kavier's catalog. A100-SXM4-80GB, A100-PCIE-80GB and A100-PCIE-40GB map to Kavier's names for
  the same part; A100-SXM4-40GB, which Kavier cannot predict, is refused when the context is built.
- A GPU count that is not a power of two gets an exact node layout (12 GPUs at 8 per node is 6 x 2), and a
  repeated grid entry is recommended once.
- Fractional epochs are accepted; `epochs` and `dataset_size` must be positive, `max_slowdown` at least 1 and
  `top_k` at least 1.
- `recommend-trace` and the dashboard keep every layout within the cluster's GPUs per node.
- `strategy.max_slowdown` applies to `recommend-job --config` and the dashboard, as it did for batch CSV input.
- `lookup: default` reads a run database shipped in the package, and `recommend-job` without a config runs the
  packaged default policy, so both work after `pip install`.
- Log records go to stderr, so `recommend-job --config` writes only the recommendation JSON to stdout.
- CI runs on Linux only, on pull requests, with a time cap on every job and `uv sync --locked`; the native-ML
  tests are a gate. Releases trigger on plain semver tags only, check the version in `pyproject.toml`,
  `CITATION.cff` and `.zenodo.json`, and the GitHub release no longer waits for the PyPI upload.
- The Docker image no longer installs git, and sets `PORTFOLIO_DIR` to a folder its user owns.
- `config/coastline_functionality/infrastructure.yaml` declares the cluster the docs and the thesis describe:
  32 NVIDIA-A100-SXM4-80GB GPUs on 4 nodes of 8. The dashboard header and `/api/infrastructure` list one GPU type.
- A data-driven model whose file is not in the install (the PyPI wheel leaves several out) fails with a message
  that says where to get the model files, and `/api/recommend` answers 404 with that message.
- Loading `tabpfn.pkl`, which scikit-learn 1.8.0 wrote, logs one INFO line with both scikit-learn versions
  instead of a scikit-learn version warning per estimator class.

- `portfolio/tabpfn.pkl` and `portfolio/random_forest.pkl` are plain git files instead of Git LFS objects, so the
  source archive and its Zenodo record contain every trained model.

### Removed

- The demo-tuned TabPFN models in `portfolio/custom/`, which took precedence over the bundled model. The
  `tabpfn` predictor now uses `portfolio/tabpfn.pkl`, the model the thesis evaluates; `custom/` is git-ignored.
- `dev/ado_plugin/`. The IBM ado integration is at
  <https://github.com/Radu-Nicolae/ado-coastline/tree/coastline-plugin/plugins/custom_experiments/coastline>.
- Docker Hub publishing, the Codecov upload, and the `docs` service of `docker-compose.yml`, whose Dockerfile
  did not exist.
- The `[tool.uv]` dependency override and the `[tool.uv.sources]` Kavier pin; neither reached the wheel.
- The Python 3.14 classifier, which `requires-python` excludes.
- The previous `docs/` pages, including `CODE_QUALITY.md` and the guides; the three-page documentation site
  replaces them.

### Fixed

- `coastline.recommend` ran the 'intelligent' predictor when the predictor name was not in lower case.
- Fractional epochs were truncated, so runtime and energy were wrong or missing.
- A GPU budget that is not a power of two was rounded up to whole nodes, giving duplicate recommendations.
- `recommend-job --config` ignored the GPU declared under `workload:`.
- `recommend_csv` failed on a row with a trailing comma and left a half-written output, left the previous output
  in place for an input with no rows, and failed on an empty config section.
- A cache hit on a model Kavier does not cover was discarded. The hit carries the GPU power measured in its run,
  which recommendations and `coastline simulate` use when the power predictor has none; the lookup calls no
  other predictor. Lookup rows with a non-numeric cell are skipped instead of failing.
- The dashboard's config file was ignored under Docker (`EXPERIMENT_CONFIG`), `/api/version` reported 0.1.0,
  queue jobs and CSV imports used different time bases, and ISO arrival times were skipped on import. A CSV of
  ISO arrival times starts at the latest arrival already queued, so its jobs queue after the ones already there.
- `plot-trace --baseline` counted multi-node jobs' GPUs per node, and `--submit-col` ignored ISO timestamps.
- Bad CLI input gives an `error:` line and exit code 2 instead of a traceback; in the interactive REPL, q and
  Ctrl-C quit and Esc goes back to the start.
- `coastline utils tune` checks that it can write its output folder before training.
- A test imported a module that no longer exists, so the native-ML tests failed.

## [0.2.2] - 2026-09-16

The thesis freeze: the state of Coastline used for the MSc thesis experiments, tagged
`v0.2.2-thesis`. The tag marks the freeze; the release workflow does not publish it.

### Added

- Parallel pipeline stages: the recommendation stages run as fork-join phases over the
  candidates, on up to `runtime.parallel_workers` processes.
- An opt-in empirical OOM guard (`predictors.empirical_oom_guard`) on top of the feasibility
  backend.
- `constants.py`, which holds the enums and defaults.
- `docs/CODE_QUALITY.md`, a code standard of 10 functional and 5 non-functional requirements.

### Changed

- The sklearn models share one generic predictor and one generic trainer.
- The freeze tag pins Kavier to its thesis tag `v0.5.1-thesis` (a git source) instead of the PyPI
  wheel `0.5.0.2`, so the reproducibility capsule runs against the code that produced the
  published results. `main` kept the `kavier>=0.5,<0.6` PyPI range.

### Removed

- The input name aliases: `coastline.recommend`, `Coastline` and `recommend_csv` take the
  `WorkloadSpec` field names.

### Fixed

- AutoConf got the per-device batch (`WorkloadSpec.batch_size`) where its `JobConfig.batch_size`
  expects the total effective batch, so on more than one GPU it judged a batch smaller by a factor
  of the GPU count. The AutoConf boundary now multiplies the batch by the GPU count.

[Unreleased]: https://github.com/atlarge-research/coastline-recommender/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/atlarge-research/coastline-recommender/compare/v0.2.2-thesis...v0.2.3
[0.2.2]: https://github.com/atlarge-research/coastline-recommender/releases/tag/v0.2.2-thesis
