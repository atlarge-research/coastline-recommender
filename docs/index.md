# Coastline

Coastline is the first scientific instrument for context-, policy-, and objective-aware recommendations of LLM
fine-tuning workloads.

## Features

[//]: # (TODO: Update links)

1. Supports a [multi-objective](todo_link) recommendation policy;
2. Supports a [min-GPU](todo_link) recommendation policy;
3. Recommends feasible configurations, using [IBM AutoConf](todo_link);
4. Makes [multi-objective recommendations](todo_link) using a [diverse set of simulation models]();
5. Can simulate [performance](todo_link) and [energy](todo_link);
6. Interfaces for everybody: [programmatic interface](todo_link), [graphical interface](),
   and [command line interface]();
7. Integrated with [IBM ado](https://research.ibm.com/blog/ado-accelerated-discovery-orchestrator-experiments) as a plugin experiment, via the [programmatic interface](6_specifications.md).

!!! tip
    Coastline is an open-source project and we encourage you to explore our [GitHub repository](https://github.com/atlarge-research/coastline-recommender). 
    We welcome contributions, feedback, and suggestions from the community. 
    If you encounter any issues or have ideas for improvements, please feel free to open an issue or submit a pull request.

## Installation

Coastline requires Python >=3.11.
Install Coastline using `pip`:

```console
pip install coastline-recommender
```

## In this documentation

[//]: # (TOOD: Update TOC)

In this documentation, you will learn how to install, configure, and use Coastline.
In this documentation you will find:

1. [Getting started](2_getting-started.md) — install Coastline and make your first recommendation.
2. [Setting up an experiment](3_experiment.md) — the config folder, file by file.
3. [Recommendation policies](4_recommendation_policies.md) — min-GPU and multi-objective.
4. [Simulation models](5_simulation_models.md) — the performance, energy, and feasibility predictors.
5. [Feasibility checker](6_feasibility_checker.md) — IBM AutoConf.
6. [Specifications](6_specifications.md) — the CLI, configuration, and SDK reference.
7. [Terminology](7_terminology.md) — one canonical term per thing.


!!! info
    Coastline is jointly backed by [AtLarge Research Group](https://www.atlarge-research.com/) and [IBM Research](https://research.ibm.com/).
    Main contributors: [Radu Nicolae](https://radu-nicolae.com) and [Daniele Lotito](https://danielelotito.github.io/dl-codespace/).