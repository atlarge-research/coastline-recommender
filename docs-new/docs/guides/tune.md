# Tune a predictor

Fit a data-driven predictor to **your** measured fine-tuning runs. Tuned models usually beat the
generic physics path on the hardware and models they were trained on.

Requires the `[ml]` extra.

```bash
coastline tune --data runs.csv --model tabpfn
```

The tuned artifact is saved to `models/` and picked up automatically wherever you select the model —
`predictors.performance: tabpfn` in a config, `predictor="tabpfn"` in Python, or
`--method tabpfn` on trace commands.

## Dataset format

One fine-tuning run per row. Print the exact contract with:

```bash
coastline tune --format
```

A CSV that doesn't meet the contract fails loudly, listing what's missing. Quality problems that
don't block tuning (too few rows, a single configuration, models unknown to the spec library) are
reported as warnings.

## Validate before you trust

By default all valid rows are used for tuning. Hold out a test split to get an error estimate:

```bash
coastline tune --data runs.csv --model tabpfn --train-percentage 0.8   # 20% holdout, reports MdAPE
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | – | Measured-runs CSV (required) |
| `--model` | `tabpfn` | Model to tune |
| `--train-percentage` | `1.0` | Fraction used for tuning; `< 1.0` reports holdout MdAPE |
| `--output` | auto | Artifact path override |
| `--seed` | `42` | Split seed |
| `--format` | – | Print the dataset contract and exit |
