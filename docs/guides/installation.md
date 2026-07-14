# Installation

Coastline is provided under three standalone tools:

- The command-line interface, for a lightweight, local version of Coastline;
- The dashboard web interface, for user-friendly interaction with Coastline recommendations;
- The Python SDK, for developers.

## Local installation

First, we recommend using `uv` for Python package management.
Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to get access to `uv` on
your system.

Coastline requires Python version **3.13**.
For local installations, use a Python virtual environment setup:

```bash
uv venv --clear --python 3.13 .venv
source .venv/bin/activate
```

Install the basic Coastline package:

```bash
uv pip install coastline-recommender
```

Additionally, we provide two extra modules, `coastline-recommender[ml]` and `coastline-recommender[plot]` for
data-driven predictors and trace plotting functionality.
You must install the two modules separately:

```bash
uv pip install coastline-recommender[ml] coastline-recommender[plot]
```

## Docker installation

If you would like to avoid setting up a Python environment locally, we also provide prebuilt Docker images ready to run
Coastline.
To get started, simply pull the latest Docker images:

```bash
docker pull radu33/coastline:coastline-ui@latest
docker pull radu33/coastline:coastline-cli@latest
```

## CLI usage

Once Coastline is installed, you can execute the command-line interface using the `coastline` binary:

```bash
coastline

# Output:
# usage: coastline <command> [options]
# 
# commands:
#   recommend        Batch-recommend GPU/node configs for a CSV of workloads (CSV in -> CSV out).
#   run              Run one config-file experiment; write a recommendation.json run artifact.
#   recommend-trace  Recommend a config for every job in a fine-tuning trace CSV.
#   plot-trace       Plot a recommended trace: cluster timeline, GPUs in use + jobs queued ([plot] extra).
#   interactive      Guided keyboard-driven REPL over the recommender.
#   tune             Tune a data-driven predictor (tabpfn) on your own measured-runs CSV ([ml] extra).
# 
# Run `coastline <command> --help` for command-specific options.
```

If you installed via Docker, run Coastline using the command below.
Any arguments appended to the command will be redirected to Coastline as if running on the local machine.

[//]: # (TODO: Update docker command)

```bash
docker run --rm coastline-cli
```

You can learn more about the CLI capabilities by reading the [command-line guide]().


## SDK usage

Coastline can also be used via its programmatic interface, which allows developers to integrate Coastline into their own
Python applications.

You can add coastline as a dependency to your `uv`-backed project as follows:

```bash
uv add coastline-recommender
```

!!! warning
    **The package name is `coastline-recommender`; the import name is `coastline`.**

    `uv pip install coastline` installs an unrelated package that also imports as `coastline`.

    Additionally, do note that your Python application must satisfy the **Python>=3.13** requirement to be compatible with `coastline-recommender`.

Once available as a dependency, you can use Coastline in your application by simply importing it:

```python
import coastline
```

You can learn more about the Python SDK in the [reference section]().

## UI usage

Once Coastline is installed, you can deploy the web interface to [localhost:8000](http://127.0.0.1:8000) using the command
below:

```bash
costline-ui

# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

You can learn more about the UI capabilities by reading the [command-line guide]().

