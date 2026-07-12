# Recommendation policies

In this section, you will learn how Coastline recommends the best-fit configuration for a workload. In this section you will find:

1. [Min-GPU recommendation](#min-gpu-recommendation) — the minimum compute power, yet still computationally feasible; currently used in IBM production pipelines. The min-GPU is feasibility-aware.
2. [Multi-objective recommendation](#multi-objective-recommendation) — weighted predictions of previously proposed or peer-reviewed simulators. The multi-objective is simulation-driven, feasibility-aware, and user-tunable.
3. [Performance-oriented recommendation](#performance-oriented-recommendation) — the multi-objective preset that prioritizes time-to-completion (β = 0.8).
4. [Energy-saver recommendation](#energy-saver-recommendation) — the multi-objective preset that prioritizes sustainability (α = 0.8).

| Policy | Preset | α (energy)                | β (performance)        | Selection effect |
|---|---|---------------------------|------------------------|---|
| `min_gpu` | — | —                         | —                      | Fewest GPUs, feasible execution |
| `multi_objective` | `balanced` (default) | 0.5                       | 0.5                    | Equal trade-off |
| `multi_objective` | `performance` | 0.2                       | 0.8                    | Prioritizes fast execution |
| `multi_objective` | `energy` | 0.8                       | 0.2                    | Prioritizes low power draw |
|`multi_objective` | `custom` | user-defined (α in [0,1]) | user-defined (β=(1-α)) | User-defined trade-off |

## 1. Min-GPU recommendation { #min-gpu-recommendation }

The Min-GPU policy targets the common enterprise goal of provisioning the *smallest* GPU allocation that can still execute the submitted fine-tuning workload without failure (e.g., out-of-memory), thereby reducing cost and failed job restarts. The origin of the Min-GPU recommendation is IBM ADO AutoConf, which uses AutoGluon validity classification on fine-tuning job configurations; Coastline only implements the functionality, using already vetted [feasibility-checking](...) functionality from real-world deployments.

```python
def recommend_min_gpu(workload, context, feasibility, simulator):
    feasible = [c for c in grid(workload, context)      # total GPUs × batch size
                if feasibility.is_feasible(c)]          # feasibility checker (AutoConf)
    for c in feasible:
        c.throughput, c.power = simulator.predict(c)
    feasible.sort(key=lambda c: (c.total_gpus, -c.throughput))
    return feasible[:top_k]                             # fewest GPUs first
```

Min-GPU ranks every feasible configuration by total GPU count, ascending; among configurations with an equal GPU count, the higher-throughput configuration wins.


## 2. Multi-objective recommendation { #multi-objective-recommendation }

The Multi-objective recommendation evaluates all feasible points in the batch × GPU grid, scores each configuration, and ranks the configurations according to a user-selected trade-off between performance and sustainability. The multi-objective policy uses the full 2D exploration space, applying the same [feasibility check](...) and simulation as Min-GPU, yet selecting based on a weighted score rather than minimum resource usage.

For each feasible configuration $n$ in the grid, Coastline computes a time cost $t_n = 1 / T_n$, the reciprocal of the predicted throughput $T_n$ (the work per token cancels in the normalization), and a power cost $p_n = W_n \times G_n$, the predicted per-GPU power draw times the total number of GPUs. Both costs are min–max normalized over the feasible grid into scores in $[0, 1]$, where higher is better:

$$
s_{\text{r}, n} = \frac{t_{\max} - t_n}{t_{\max} - t_{\min}},
\qquad
s_{\text{e}, n} = \frac{p_{\max} - p_n}{p_{\max} - p_{\min}}
$$

$$
S_n = \alpha \times s_{\text{e}, n} + \beta \times s_{\text{r}, n},
\qquad \alpha + \beta = 1,\; \alpha, \beta \ge 0
$$

where $s_{\text{r}, n}$ and $s_{\text{e}, n}$ denote the score of configuration $n$, for runtime (r) and energy (e), respectively, and $t_{\min}, t_{\max}, p_{\min}, p_{\max}$ are the smallest and largest time and power costs among the feasible configurations. The final ranking score combines both objectives through a weighted sum. The higher the score, the better the configuration. The `-frontier` preset variants (`balanced-frontier`, `performance-frontier`, `energy-frontier`) drop dominated configurations before the normalization.

The weights α and β are controlled by the user through a *preset* selection, as presented in the table above. For the `balanced` preset, the recommender system weights performance and energy consumption equally and aims to find, through a grid search, the best-fit configuration (i.e., the best score, where higher is better).


## 3. Performance-oriented recommendation { #performance-oriented-recommendation }

Derived from the multi-objective policy, the performance-oriented preset uses a β=0.8 (the performance weight in the weighting function) and a remaining α=0.2 (the energy weight), although four times lower, still playing a role in the final recommendations. This preset targets stakeholders who prioritize time-to-completion over sustainability, while still accounting for responsible computing by assigning a small yet significant weight to energy consumption.


## 4. Energy-saver recommendation { #energy-saver-recommendation }

The energy-saver recommendation is complementary to the multi-objective recommendation. For the energy-saver recommendation, the energy weight dominates (α = 0.8), while the performance weight remains β = 0.2. The weighted combined score is largely influenced by energy efficiency while still ensuring that the jobs are completed within a reasonable amount of time. This preset targets stakeholders who prioritize sustainability over execution speed, such as for non-urgent workload execution.
