# Coastline functionality

Configs the code loads to run, and two example inputs. Edit with care.

- **experiment.yaml**: the recommendation policy (strategy, predictors, grid) that `coastline recommend-job` and the
  dashboard use when no other config is given. Its `workload` block is the job that `coastline recommend-job` runs.
- **infrastructure.yaml**: the cluster's GPUs (total, per node, nodes). The CLI and the dashboard keep every
  recommendation within them. Edit with care.
- **sample_workloads.csv**: example input for `coastline recommend-job --input ... --output ...`.
- **sample_trace.csv**: example fine-tuning trace for `coastline recommend-trace`.
