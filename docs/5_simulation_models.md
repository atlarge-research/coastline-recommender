# Simulation models

In this section, you will learn how Coastline predicts the performance and energy of LLM fine-tuning workloads. In this section you will find:

1. [Performance models](#performance-models) — the 12 predictive techniques, of retrieval-based, analytical, and data-driven natures, predicting throughput and runtime.
    1. [The intelligent mode of operation](#the-intelligent-mode-of-operation) — query the run database first; predict only on a miss.
    2. [Cache lookup](#cache-lookup) — exact-match retrieval from the run database.
    3. [Kavier](#kavier) — the analytical predictor: step runtime first, throughput derived.
    4. [Data-driven models](#data-driven-models) — ten ML models trained on the IBM profiling trace.
2. [Energy prediction](#energy-prediction) — power draw from Kavier, in the same engine call.
3. TODO: co2 models

## 1. Performance models { #performance-models }

The models are of various natures: retrieval-based (PR), analytical (PA), or data-driven (PD). For the identification and model type, we give unique identifiers and encode the type of the model as P = performance predictor, R = cache-retrieval, A = analytical, D = data-driven. For the metrics, T = throughput (tokens/s), R = runtime.

Each predictive model receives eight inputs:
(1) LLM name,
(2) GPU model,
(3) number of nodes,
(4) fine-tuning method,
(5) batch size,
(6) tokens per sample,
(7) GPUs per node,
and (8) number of training epochs.

Each predictive model produces two outputs:
(1) the training throughput, measured in tokens per second, and
(2) the total training runtime, measured in seconds (or derived metrics).

!!! note
To predict performance and ensure compatibility with the modular software architecture, each model, independent of its nature, receives the same eight (8) inputs and gives the same two (2) outputs.

| ID | Type | Model | Config name | Predicts | Description |
|---|---|---|---|---|---|
| 1 | PR | Cache Lookup | `cache` | T, R | Exact-match retrieval from run database |
| 2 | PA | Kavier | `kavier` | T, R | $T = \tfrac{B \times S \times G_n}{t_s}$, &nbsp; $t_s = \tfrac{6 \times P_a \times B \times S}{F \times \eta} + \tfrac{20 \times P_t}{B_m} + t_c$ |
| 3 | PD | Random Forest | `random_forest` | T, R | Bagged ensemble of decision trees |
| 4 | PD | XGBoost | `xgboost` | T, R | Gradient-boosted decision trees |
| 5 | PD | LightGBM | `lightgbm` | T, R | Histogram-based gradient boosting |
| 6 | PD | CatBoost | `catboost` | T, R | Ordered boosting with categorical features |
| 7 | PD | Bayesian Ridge | `bayesian_ridge` | T, R | Bayesian linear regression with learned priors |
| 8 | PD | SVR | `svr` | T, R | Kernel-based support vector regression |
| 9 | PD | KNN | `knn` | T, R | Prediction by averaging $k$ nearest neighbors |
| 10 | PD | Gaussian Process | `gaussian_process` | T, R | Bayesian non-parametric regression |
| 11 | PD | Deep Learning | `deep_learning` | T, R | Multi-layer fully connected neural network |
| 12 | PD | TabPFN | `tabpfn` | T, R | Prior-fitted transformer for tabular data |

*Symbols (Kavier, ID=2): $t_s$ = step time; $P_a$, $P_t$ = active, trainable parameters; $B$ = batch size; $S$ = sequence length; $G_n$ = GPUs per node; $F$ = GPU peak FLOP/s; $\eta$ = MFU; $B_m$ = memory bandwidth; $t_c$ = network overhead.*

The default, `intelligent`, composes models 1 and 2: [cache](#cache-lookup), then a fallback simulation model — [Kavier](#kavier) by default, or any trained model set with `predictors.fallback`.

### 1.1 The intelligent mode of operation { #the-intelligent-mode-of-operation }

We identify, propose, and implement an *"intelligent mode of operation"* for Coastline: what if, instead of predicting, which is error-prone (no model is perfect) and resource-consuming, the tool would extract a previous measurement from a database?

The intelligent mode of operation follows a strategy whereby the system first queries the database for previous measurements:

- *if entry available in the database:* retrieve, in O(1) time complexity, the present entry (or an aggregated function of multiple entries, if more are present) to the user. Each entry is a previously monitored scenario in which a user of the system deployed a workload on the infrastructure, and the system automatically monitored metrics such as throughput, per-step runtime, and energy consumption. The cycle of (1) deploy, (2) monitor, (3) store metadata creates a snowball effect where the database grows with the usage of the tool.
- *else (if entry not available in the database):* the fallback simulation model makes the prediction — the [Kavier](#kavier) analytical model by default, or any trained model chosen with `predictors.fallback`.

### 1.2 Cache lookup { #cache-lookup }

The cache-lookup high-level approach has been used in computer systems since the 1960s, with the invention and adoption of caching, across various scopes, including job scheduling strategies or runtime estimation. For LLM fine-tuning simulation, we propose a model following an analogous strategy.

Coastline uses a large-scale dataset comprising over 30,000 configurations deployed on IBM infrastructure, profiled, and aggregated. While a cartesian product of hundreds of possible LLMs run with a large number of infrastructure configurations can easily overtake the number of grains of sand on Earth, we identify hotspots in this massive exploration space: while tens of thousands of LLMs exist, only a few tens are used at large; similarly for GPUs, with only a few GPUs used at large for training or fine-tuning LLMs, such as NVIDIA's A100 and H100.

### 1.3 Kavier { #kavier }

As an analytical predictor, Kavier's predictions are based on properties of the fine-tuning experiment. At the core, Kavier first computes the step runtime and then derives the throughput.

Kavier computes the time of one training step with a four-term sum: the forward pass, the backward pass, the optimizer update, and the gradient synchronization across GPUs. When gradient accumulation is used, the forward and backward passes are repeated $G_a$ times before a single optimizer update and gradient all-reduce:

$$
T_s = G_a \times (T_f + T_b) + T_o + T_c
$$

where $T_s$ is the step time in seconds, $G_a$ is the number of gradient-accumulation micro-steps (default 1), $T_f$ is the forward pass time, $T_b$ is the backward pass time, $T_o$ is the optimizer update time, and $T_c$ is the gradient communication time.

To compute throughput, Kavier derives the final result from (1) step runtime (amount of seconds per step) and (2) the amount of tokens per step, thus obtaining (3) the amount of tokens per second. The physics-based model currently does not consider real-world phenomena, such as framework overhead, batching delays, or differences between models and methods; Kavier therefore applies three multiplicative calibration scales — $c_m$ (per fine-tuning method), $c_o$ (per LLM), and $c_i$ (per (model, method, GPU, $G$) interaction) — together with one divisive multi-GPU correction $m_g$:

$$
T = \frac{G_a \times B \times S \times G_n}{m_g \times T_s} \times c_m \times c_o \times c_i
$$

where $T$ is throughput in tokens per second, $B$ is micro-batch size, $S$ is sequence length in tokens, $G_n$ is GPUs per node, $T_s$ is the step time from the equation above, $m_g$ is a calibrated multi-GPU correction, $c_m$ is a per-method calibration scale, $c_o$ is a per-LLM calibration scale, and $c_i$ is a per-(model, method, GPU, $G$) interaction calibration scale.

### 1.4 Data-driven models { #data-driven-models }

We design all data-driven models to be trained on varied datasets, with values that span orders of magnitude (e.g., outputs ranging from 100s to 100,000s of tokens per second). All targets are trained in log space and then exponentiated before prediction, with a 70-15-15 train-validation-test split and an identical seed, to ensure reproducibility.

| Model | Trade-off |
|---|---|
| `random_forest` | Rapid, with acceptable accuracy, yet not on par with gradient boosting or TabPFN; aims for consistency rather than peak accuracy |
| `xgboost` | Prioritizes accuracy through boosting and L1/L2 regularization; training cost comparable to random forest thanks to early stopping |
| `lightgbm` | Accuracy close to XGBoost, with a lower training cost than deep models, but below TabPFN |
| `catboost` | Captures non-linear interactions better than linear or distance-based models, at the cost of longer training and larger models |
| `bayesian_ridge` | Lightweight; trains and predicts quickly, yet risks underfitting highly variable and non-linear effects |
| `svr` | Training and inference time scale exponentially with the size of the training dataset; lower interpretability and explainability |
| `knn` | Simple and fast to query; prediction quality drops when no close neighbors are available (e.g., rare model-GPU pairs) |
| `gaussian_process` | Uses information from the entire training set and is more robust to outliers; training and storage costs increase rapidly as the dataset grows |
| `deep_learning` | Represents interactions between embeddings and numeric fields in a single function, at the cost of longer training and a dependence on PyTorch |
| `tabpfn` | The strongest-performing model in our benchmark; orders of magnitude slower than the other models |

## 2. Energy prediction { #energy-prediction }

Coastline predicts power draw with `kavier_power`, the energy counterpart of the Kavier analytical model. Kavier returns power alongside throughput in a single engine call, so the energy prediction adds no second simulation — the pipeline reuses the same call for both metrics.
