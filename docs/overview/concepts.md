# Concepts

Coastline is an LLM recommendation system using context-, objective, and policy-aware simulation of (batches of) workloads.
On this page we define and explain every keyword above.

[//]: # (TODO: add links)

## Recommendation

Coastline receives an input entity to be improved. 
Recommendation is the process of improving the entity to better align with the established objectives. 
The output of this recommendation is a set of improved entities.

!!! example
    One can provide to Coastline the goal of minimizing the runtime of an LLM workload and a cluster configuration on which the given workload should be executed.
    Coastline will recommend the best GPU configuration to use for the received input.

## Recommendation policy 

Recommendation policies instruct Coastline's pipeline to structure recommendations in such a way that best represents a given [objective](#objective-awareness).

!!! example
    _Multi-objective performance-oriented_, instructs Coastline's pipeline to favor the configuration with the best performance.

## Simulation

Simulation is the behind-the-scenes engine that drives recommendations.
Coastline predicts a given metric of interest given a batch of inputs by means of simulation.
Below are the main use-cases of simulation inside Coastline:

1. multi-objective recommendation policy
2. show the performance/sustainability of the selected metrics

!!! example
    - Input: 1 node x 4 A100, batch=64, mistral-7b, for 3 epochs
    - Output: 34 min runtime, 582 Wh

## Context-awareness

The context mimics the real-world computer infrastructure to influence the final Coastline recommendation.
You can understand context-awareness as the set of hardware/logical components that make up a datacenter.

!!! example
    64 nodes x 8 A100 GPUs, with 100 Gbps network

## Objective-awareness

The objective is the goal for which Coastline makes recommendations. 
In a datacenter environment, a (service-level) objective influences a decisions (e.g., up-scale infrastructure, change configuration); for Coastline, the objectives influence the configuration recommendation.

!!! example
    The objective of _energy_ influences Coastline to prefer energy-efficient configurations.

## Policy-awareness

Coastline produces policy-aware recommendations based on a [recommendation policy](#recommendation-policy) configured by the user in the experiment setup.

## Fine-tuning job

A fine-tuning job contains specifications about the LLM to be tuned, about the GPUs (/infrastructure used), about the dataset to be used, and about the job itself (e.g., number of epochs, batch size).

!!! example
    A fine-tuning job for the LLM model `mistral-7b` with 64 GPUs, a batch size of 128, and 3 epochs.

## Workload

A workload is a set of [fine-tuning jobs](#fine-tuning-job) to be patched with Coastline recommendations. 
A workload can contain one or more jobs.

!!! example 
    A workload can be a set of 10 fine-tuning jobs for different LLMs, each. 
    Or, a workload can be 1 fine-tuning job.

