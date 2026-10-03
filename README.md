# Coastline

A context-aware recommender system for fine-tuning LLMs.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/atlarge-research/coastline-recommender/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-site-green.svg)](https://atlarge-research.github.io/coastline-recommender/)
[![PyPI](https://img.shields.io/pypi/v/coastline-recommender.svg)](https://pypi.org/project/coastline-recommender/)

Coastline makes context-, objective-, and policy-aware infrastructure recommendations for LLM fine-tuning
workloads. It accounts for infrastructure constraints, workload demands, and user objectives, and recommends
best-fit configurations as part of an LLM fine-tuning ecosystem. Coastline predicts performance with
physics-driven simulation (Kavier) and machine-learning models. Every candidate configuration is cross-checked
through a feasibility module (IBM AutoConf) before being output to the user.

Built with PriorLabs-TabPFN. See the TabPFN section below.

## Install

```bash
pip install coastline-recommender          # Kavier, AutoConf, the CLI, and the dashboard
pip install "coastline-recommender[ml]"    # adds the data-driven predictors
```

Python 3.11 to 3.13. The import name is `coastline`.

## Quick start

From a clone of this repository:

```bash
uv sync
uv run coastline recommend-job
uv run coastline explain --model mistral-7b-v0.1 --method lora --gpu-model NVIDIA-A100-SXM4-80GB \
  --tokens 2048 --batch-size 8
uv run coastline --help
```

`coastline recommend-job` recommends a configuration for the job declared in
`config/coastline_functionality/experiment.yaml`. The subcommands are `recommend-job`, `recommend-trace`,
`simulate`, `explain`, and `utils`. Each documents its flags with `--help`. `uv run coastline-ui` serves the
dashboard at <http://127.0.0.1:8000>.

## Documentation

<https://atlarge-research.github.io/coastline-recommender/>. Build it locally with
`uv run --group docs mkdocs serve`.

## Development

`uv sync` installs the dev tools. CI runs these gates on every pull request
([ci.yml](https://github.com/atlarge-research/coastline-recommender/blob/main/.github/workflows/ci.yml)):

```bash
uv run pre-commit run --all-files --show-diff-on-failure      # ruff check, ruff format, whitespace
uv run mypy                                                   # strict, on the packages listed in pyproject.toml
uv run --all-extras pytest --cov
uv run --all-extras pytest -m ml_isolated -p no:cacheprovider FILE  # native ML backends, one file per process
uv run --group docs mkdocs build --strict
```

Run `uv run pre-commit install` once per clone to get the hooks on commit.

## TabPFN

Built with PriorLabs-TabPFN.

The `tabpfn` predictor and the model file `portfolio/tabpfn.pkl` (in
`src/coastline/sdk/predictors/performance/data_driven/`) contain TabPFN v2 weights. TabPFN v2 is described in
Hollmann et al., "Accurate predictions on small data with a tabular foundation model", Nature 637, 319-326
(2025), [doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6).
The weights are licensed under the Prior Labs License v1.2; a copy is in
[LICENSE-TabPFN.txt](https://github.com/atlarge-research/coastline-recommender/blob/main/LICENSE-TabPFN.txt). The
model file is stored with Git LFS and left out of the PyPI wheel; clone the repository with Git LFS to use it.

## Citation

See [CITATION.cff](https://github.com/atlarge-research/coastline-recommender/blob/main/CITATION.cff).

## License

MIT. See [LICENSE](https://github.com/atlarge-research/coastline-recommender/blob/main/LICENSE). The TabPFN weights
are under the Prior Labs License v1.2, in
[LICENSE-TabPFN.txt](https://github.com/atlarge-research/coastline-recommender/blob/main/LICENSE-TabPFN.txt).
