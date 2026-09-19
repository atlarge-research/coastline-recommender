# Feasibility checker

In this section, you will learn how Coastline verifies configuration feasibility. In this section you will find:

1. [Where the feasibility checker runs](#where-the-feasibility-checker-runs) — the pipeline stage that filters the grid before any prediction.
2. [The backends](#the-backends) — the three values of `predictors.feasibility`.
    1. [AutoConf](#autoconf) — the structural guards plus the IBM AutoConf classifier.
    2. [Rules](#rules) — the structural guards alone.
3. [The empirical OOM guard](#the-empirical-oom-guard) — the opt-in per-device token budget.

## 1. Where the feasibility checker runs { #where-the-feasibility-checker-runs }

The pipeline generates the candidate configurations, checks each candidate for feasibility, and predicts throughput and power only for the candidates that pass. An infeasible candidate never reaches the [recommendation policy](4_recommendation_policies.md), so every configuration Coastline recommends is a configuration the feasibility checker admitted.

Select the checker with the `feasibility` key of the [`predictors:` block](3_experiment.md#config-predictors), or with the `--feasibility` flag of [`coastline simulate`](guides/cli.md#simulate), [`coastline explain`](guides/cli.md#explain) and [`coastline recommend-trace`](guides/cli.md#recommend-a-trace).

```yaml
predictors:
  feasibility: "autoconf"        # autoconf (default) | rules | none
  empirical_oom_guard: false     # the opt-in per-device token budget, off by default
```

## 2. The backends { #the-backends }

| Config name | Checks | Needs AutoConf |
|-------------|--------|----------------|
| `autoconf` | the structural guards, then the AutoConf validity classifier | yes |
| `rules` | the structural guards alone | no |
| `none` | nothing: every candidate is feasible | no |

The [empirical OOM guard](#the-empirical-oom-guard) layers over whichever backend the config selects.

### 2.1 AutoConf { #autoconf }

`autoconf` is the default and the only backend with a memory model. The checker runs the [structural guards](#rules) first, then hands every candidate the guards admit to IBM AutoConf — the `ado-autoconf` plugin of IBM's [ado](https://research.ibm.com/blog/ado-accelerated-discovery-orchestrator-experiments) orchestrator, which pairs a rule-based classifier with a validity model trained on measured fine-tuning runs and answers one question per candidate: does this configuration run, or does it exhaust GPU memory?

AutoConf's `JobConfig.batch_size` is the total effective batch, while Coastline's `batch_size` is per device, so Coastline multiplies the per-device batch by the GPU count at the boundary. The conversion also satisfies AutoConf's own rule-based classifier, which requires the effective batch to divide by the GPU count.

A candidate AutoConf cannot decide — an invalid job configuration, or a failed prediction — counts as infeasible, so the sweep continues over the rest of the grid. AutoConf ships in the core install; without it, `autoconf` refuses to run unless `COASTLINE_ALLOW_RULES_FALLBACK=1` knowingly degrades the run to `rules`.

### 2.2 Rules { #rules }

`rules` applies the structural guards alone: a candidate is feasible when the GPU count is at least 1 and the per-device batch size is at least 1. The guards are two integer comparisons — **no memory model and no OOM check** — so the `rules` backend admits every configuration that describes a job at all.

`batch_size` is per device (Kavier's convention), so a per-device batch has nothing left to divide by the GPU count: the earlier divisibility rule was removed when the field became per device.

Use `rules` for a cheap sweep, or for an environment without AutoConf. To keep a memory-shaped veto on that path, enable the [empirical OOM guard](#the-empirical-oom-guard).

## 3. The empirical OOM guard { #the-empirical-oom-guard }

The empirical OOM guard rejects a candidate whose per-device token load — the per-device batch size times the tokens per sample — exceeds a budget of 60,224 tokens per device, fitted to the OOMs observed in Coastline's calibration campaigns.

The guard is **opt-in**: set `predictors.empirical_oom_guard: true` in the config, or pass `empirical_oom_guard=True` to `Coastline(...)` through the programmatic interface. The guard then layers over whichever backend the config selects and runs first, so a candidate the guard rejects never pays for a classifier call. The guard can only veto: a candidate the guard admits is decided entirely by the backend.

```yaml
predictors:
  feasibility: "rules"
  empirical_oom_guard: true      # the per-device token budget, over the structural guards
```

Treat the budget as a guard, not as a classifier: a single threshold on the per-device token load, selected on the same campaigns it is measured against.
