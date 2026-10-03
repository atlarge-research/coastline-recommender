"""Empirical OOM guard: a per-device token budget derived from measured campaigns.

AutoConf errs both ways. It accepts configurations far outside its training data (the largest
valid per-device load observed was 131,072 tokens; the recommender asked for 633,600 and the
cluster ran out of memory), and it rejects some models because of a failure prior learned from
its training data rather than a memory calculation.

The campaigns support a simpler rule that catches most of what AutoConf missed:

    per_device_batch_size x max_seq_length  >  threshold   ->  infeasible

The guard is opt-in and can only turn a feasible verdict into an infeasible one. It wraps the
backend the config selected, so ``autoconf`` still does the OOM classification.
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

    ``WorkloadSpec.batch_size`` is per device (Kavier's convention) and ``tokens_per_sample`` is
    the max sequence length, so their product is the tokens one device must hold. The observed
    OOMs track this product; they do not track the effective batch.
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

    The guard is cheap arithmetic and the backend may be an AutoGluon model, so a config the
    guard rejects costs no classifier call. The guard can only veto; the backend decides every
    config the guard passes.
    """

    def __init__(self, guard: TokenBudgetFeasibilityChecker, backend: _Checker) -> None:
        self._guard = guard
        self._backend = backend

    @property
    def EXPENSIVE(self) -> bool:  # noqa: N802 (mirrors the class-level flag on plain checkers)
        """Same as the wrapped backend; the guard itself is arithmetic."""
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

    def batches(self) -> bool:
        """Whether the backend batches; the guard itself runs per candidate."""
        backend_batches = getattr(self._backend, "batches", None)
        return bool(backend_batches()) if backend_batches is not None else False

    def check_chunk(self, workloads: Sequence[WorkloadSpec]) -> list[tuple[bool, dict[str, Any]]]:
        """Guard every candidate, then pass the survivors to the backend in one call.

        When the guard is enabled it is the outermost checker, so without this method the backend
        could not batch.
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
        if len(backend_results) != len(survivors):  # pragma: no cover - backends assert this themselves
            raise RuntimeError(
                f"the feasibility backend returned {len(backend_results)} verdicts for {len(survivors)} candidates"
            )
        for position, guard_metadata, (ok, backend_metadata) in zip(positions, guard_metas, backend_results):
            results[position] = (ok, {**guard_metadata, **backend_metadata})
        return results
