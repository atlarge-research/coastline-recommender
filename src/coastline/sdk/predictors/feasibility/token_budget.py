"""The empirical OOM guard: a per-device token-budget ceiling derived from measured campaigns.

AutoConf fails in both directions. It waves through configurations far beyond its training
envelope (the largest validly observed per-device load was 131,072 tokens; the recommender asked
for 633,600 and the cluster OOMed), and it rejects some models on a training-data failure prior
rather than a memory calculation.

The campaigns support a blunter rule that catches most of what AutoConf missed:

    per_device_batch_size x max_seq_length  >  threshold   ->  infeasible

This checker is a *guard*, never a replacement: it only ever turns a feasible verdict into an
infeasible one, and it is opt-in. It wraps whatever backend the config already selected, so
``autoconf`` keeps doing the OOM classification and this adds the envelope ceiling on top.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, Sequence

from coastline.sdk.models.workload import WorkloadSpec

logger = logging.getLogger(__name__)


class _Checker(Protocol):
    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]: ...


class TokenBudgetFeasibilityChecker:
    """Reject a configuration whose per-device token load exceeds ``threshold``.

    ``WorkloadSpec.batch_size`` is per-device (Kavier's convention) and ``tokens_per_sample`` is
    the max sequence length, so their product is the tokens one device must hold. That product,
    not the effective batch, is what tracks the observed OOMs.
    """

    #: Arithmetic on two integers; never worth a worker dispatch on its own.
    EXPENSIVE = False

    def __init__(self, threshold: int) -> None:
        if threshold < 1:
            raise ValueError(f"token-budget threshold must be >= 1, got {threshold}")
        self.threshold = threshold

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        tokens_per_device = workload.batch_size * workload.tokens_per_sample
        metadata: dict[str, Any] = {
            "guard": "token_budget",
            "tokens_per_device": tokens_per_device,
            "token_budget_threshold": self.threshold,
        }
        if tokens_per_device > self.threshold:
            metadata["reason"] = (
                f"per-device token load {tokens_per_device} "
                f"(batch {workload.batch_size} x {workload.tokens_per_sample} tokens) "
                f"exceeds the empirical budget of {self.threshold}"
            )
            return False, metadata
        return True, metadata


class GuardedFeasibilityChecker:
    """Run the empirical guard first, then the selected backend.

    Guard-first is deliberate: the guard is cheap arithmetic and the backend may be an AutoGluon
    model, so a config the guard already rejects never pays for a classifier call. The guard can
    only veto; a config it passes is decided entirely by the backend.
    """

    def __init__(self, guard: TokenBudgetFeasibilityChecker, backend: _Checker) -> None:
        self._guard = guard
        self._backend = backend

    @property
    def EXPENSIVE(self) -> bool:  # noqa: N802 — mirrors the class-level flag on plain checkers
        """Worth forking exactly when the wrapped backend is: the guard itself is arithmetic."""
        return bool(getattr(self._backend, "EXPENSIVE", False))

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict[str, Any]]:
        ok, guard_metadata = self._guard.is_feasible(workload)
        if not ok:
            logger.debug("token-budget guard vetoed a candidate: %s", guard_metadata.get("reason"))
            return False, guard_metadata
        ok, backend_metadata = self._backend.is_feasible(workload)
        # Keep both: the caller can see the guard ran and what headroom the config had.
        merged = {**guard_metadata, **backend_metadata}
        return ok, merged

    def check_chunk(self, workloads: Sequence[WorkloadSpec]) -> list[tuple[bool, dict[str, Any]]]:
        """Guard every candidate, then hand the survivors to the backend in one go.

        Without this the guard would hide any batching the backend offers, since it is the
        outermost checker whenever the empirical OOM guard is enabled.
        """
        results: list[Any] = [None] * len(workloads)
        survivors: list[WorkloadSpec] = []
        guard_metas: list[dict[str, Any]] = []
        positions: list[int] = []
        for position, workload in enumerate(workloads):
            ok, guard_metadata = self._guard.is_feasible(workload)
            if not ok:
                logger.debug("token-budget guard vetoed a candidate: %s", guard_metadata.get("reason"))
                results[position] = (False, guard_metadata)
                continue
            survivors.append(workload)
            guard_metas.append(guard_metadata)
            positions.append(position)

        backend_chunk = getattr(self._backend, "check_chunk", None)
        backend_results = (
            backend_chunk(survivors) if backend_chunk is not None else [self._backend.is_feasible(w) for w in survivors]
        )
        for position, guard_metadata, (ok, backend_metadata) in zip(positions, guard_metas, backend_results):
            results[position] = (ok, {**guard_metadata, **backend_metadata})
        return results
