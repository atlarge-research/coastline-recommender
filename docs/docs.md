Guidelines
The documentation should be written in the style UV docs is written, matching the format, the abstraction level. To aid high-quality documentation, we set a set of guidelines for both the human and for Agentic-AI contributors, for structure and writing itself.


## Structure and documentation guidelines

1. Aim to write documentation across multiple levels of abstraction, and following a natural delivery flow, similarly to a scientific research article.
2. Follow the delivery format of already state-of-the-art documentations in the community, such as the documentation of UV: https://docs.astral.sh/uv/.
3. At the highest level of abstraction, follow a structure "why-what-how". For example, when proposing a new recommendation approach, what are you proposing, and how does it work.
4. Per "thing" (e.g., workload, recommendation policy), use one canonical term and never synonyms. Repeated text is better than ambiguous text. Where applicable, add hyperlinks to other sections of the documentation where the term is defined.
5. Docs change in the same PR as the behaviour they describe. Otherwise, Coastline would face documentation debt.
6. For shown commands, make sure both the input and the output are visible, both with sample text and with the format.


## Writing guidelines

Follow a short and concise writing style and adhere to scientific and technical writing guidelines. A good example are the guidelines from Strunk and White - Elements of style, from which we highlight:

1. Write in imperative.
2. Use the active voice.
3. Omit needless words.
4. Use definite, specific, concrete language.
5. Put statements in positive form.
6. Express coordinate ideas in similar form.
7. Do not have any undetermined pronouns (e.g., there, they, it). Prefer repetition of the subject to avoid ambiguity.


[//]: # (Page introduction)



# Usage

Coastline recommendation functionality can be adopted in multiple ways, depending on the user's needs and preferences. The following sections provide an overview of the different usage options available.

## Recommendations

Recommend a configuration for a single workload, in Python:

```python
import coastline

rec = coastline(throughput_estim="kavier")
results = rec({"llm_model": "mistral-7b-v0.1", "fine_tuning_method": "lora",
               "gpu_model": "NVIDIA-A100-SXM4-80GB", "tokens_per_sample": 1024, "batch_size": 32},
              total_gpus=[1, 2, 4, 8], preset="balanced")
print(results[0])
```

The first result is the best-ranked `Recommendation`:

```console
gpus_per_node=2 number_of_nodes=1 total_gpus=2 strategy='multi_objective_balanced'
predicted_throughput=7710.76 metadata={'predicted_power_watts': 223.04,
'combined_score': 0.718, 'rank': 1, 'selection_policy': 'balanced', 'batch_size': 64, ...}
```

See [first steps](...).

## Batch recommendations

Recommend configurations for a CSV of workloads, one per row:

```text
# workloads.csv
model_name,method,gpu_model,tokens_per_sample,batch_size
mistral-7b-v0.1,lora,NVIDIA-A100-SXM4-80GB,1024,16
granite-3.3-8b,full,NVIDIA-A100-SXM4-80GB,4096,4
```

```console
coastline recommend-job --config config/batch_config.yaml --input workloads.csv --output recommendations.csv
```

Each row gains the recommended configuration, its predictions, and a rationale:

```text
# recommendations.csv (excerpt)
model_name,...,recommended_total_gpus,recommended_batch_size,predicted_throughput,predicted_power_watts,feasible,rationale
mistral-7b-v0.1,...,8,32,37577.5,220.8,True,"8 GPUs (8×1, batch 32) picked for the best throughput-vs-energy balance, 4% faster than the runner-up (8 GPUs, batch 16)."
```

See the [batch guide](...).

## Traces

Annotate a fine-tuning trace with a recommendation per job, then plot the cluster timeline it produces:

```console
coastline recommend-trace --input trace.csv --output recommended.csv
coastline utils plot-trace --input recommended.csv --output timeline.pdf
```

See the [traces guide](...).

## Tuning

Fit a predictor to your own measured runs; a model tuned on your hardware beats the bundled ones:

```console
coastline utils tune --data runs.csv --model tabpfn
```

See the [tuning guide](...).

## Dashboard

Explore recommendations from the browser:

```console
coastline-ui
```

See the [dashboard guide](...).


