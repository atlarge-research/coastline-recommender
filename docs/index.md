# Coastline

Coastline is a context-aware recommender system for fine-tuning LLMs[^msc]. It accounts for infrastructure
constraints, workload demands, and user objectives, and recommends best-fit configurations as part of an LLM
fine-tuning ecosystem.

LLM fine-tuning workloads are complex and, until now, unpredictable, making it difficult to estimate computational
demands. For each workload, Coastline derives a grid of possible configurations from the tunable parameters (the
number of GPUs and the batch size), filters out infeasible configurations, predicts the throughput and power of
each feasible configuration, and ranks them by the user's goal.

Built with PriorLabs-TabPFN. See [TabPFN](#tabpfn).

## Install

```bash
pip install coastline-recommender          # Kavier, AutoConf, the CLI, and the dashboard
pip install "coastline-recommender[ml]"    # adds the data-driven predictors
coastline --help
```

Python 3.11 to 3.13. The import name is `coastline`.

## Pages

* [Usage](usage.md): the CLI, the Python API, and the dashboard.
* [How a recommendation is made](recommendation.md): grid, feasibility, prediction, and ranking.

## IBM ado

Coastline is integrated with IBM ado, an accelerated discovery orchestrator, as a plug-in. The plug-in exposes two
experiments on recommending LLM fine-tuning workloads, one using the multi-objective recommender and the other
using the min-GPU recommender. It uses the public SDK of Coastline, the same entry point as the CLI and the
dashboard. Source:
[github.com/Radu-Nicolae/ado-coastline](https://github.com/Radu-Nicolae/ado-coastline/tree/coastline-plugin/plugins/custom_experiments/coastline).

## TabPFN

Built with PriorLabs-TabPFN.

The `tabpfn` predictor and the model file `portfolio/tabpfn.pkl` contain TabPFN v2
weights. TabPFN is a Tabular Prior-Fitted Network, a transformer-style model pre-trained on synthetic tabular
tasks[^tabpfn]. The weights are licensed under the Prior Labs License v1.2; a copy is in
[LICENSE-TabPFN.txt](https://github.com/atlarge-research/coastline-recommender/blob/main/LICENSE-TabPFN.txt).

## Cite

```bibtex
@software{coastline,
  author = {Nicolae, Radu and Lotito, Daniele and Iosup, Alexandru},
  title  = {Coastline: Exploring the impact of multi-objective, context-aware recommenders on performance and
            sustainability of datacenters under LLM fine-tuning workloads},
  year   = {2026},
  url    = {https://github.com/atlarge-research/coastline-recommender}
}
```

The metadata is in [`CITATION.cff`](https://github.com/atlarge-research/coastline-recommender/blob/main/CITATION.cff).

[^msc]: R. Nicolae, D. Lotito, A. Iosup. *Coastline: Exploring the impact of multi-objective, context-aware
    recommenders on performance and sustainability of datacenters under LLM fine-tuning workloads.* MSc thesis,
    Vrije Universiteit Amsterdam, 2026.
[^tabpfn]: N. Hollmann et al. *Accurate predictions on small data with a tabular foundation model.* Nature 637,
    319-326, 2025. [doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6).
