"""The empirical OOM guard: a per-device token ceiling layered over the selected backend.

The threshold and the strict comparison are not arbitrary; they were re-derived from the nine
Zurich campaigns (135 jobs: 49 OOM, 86 completed). These tests pin the properties that derivation
established, so a later "tidy-up" cannot quietly change the semantics.
"""

from __future__ import annotations

import pytest

from coastline.sdk.constants import EMPIRICAL_OOM_TOKEN_BUDGET
from coastline.sdk.models.workload import WorkloadSpec
from coastline.sdk.pipeline.feasibility import create_feasibility_checker
from coastline.sdk.predictors.feasibility.token_budget import (
    GuardedFeasibilityChecker,
    TokenBudgetFeasibilityChecker,
)


def _workload(batch_size: int, tokens_per_sample: int) -> WorkloadSpec:
    return WorkloadSpec(
        llm_model="granite-3.1-8b-instruct",
        fine_tuning_method="full",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        gpus_per_node=8,
        number_of_nodes=1,
    )


class _Spy:
    """A backend that records whether it was consulted."""

    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.calls = 0

    def is_feasible(self, workload: WorkloadSpec) -> tuple[bool, dict]:
        self.calls += 1
        return self.verdict, {"backend": "spy"}


# --------------------------------------------------------------------------- #
# The threshold itself
# --------------------------------------------------------------------------- #


def test_the_documented_threshold_is_the_one_derived_from_the_campaigns() -> None:
    assert EMPIRICAL_OOM_TOKEN_BUDGET == 60_224


def test_the_comparison_is_strictly_greater_than() -> None:
    """Load-bearing: 14 campaign jobs sat at exactly 60,224 tokens/device and all COMPLETED.
    Using >= here would reject them and drop the rule's accuracy from 89.6% to 79.3%."""
    checker = TokenBudgetFeasibilityChecker(EMPIRICAL_OOM_TOKEN_BUDGET)

    at_threshold, _ = checker.is_feasible(_workload(32, 1882))  # 32 x 1882 == 60,224 exactly
    just_over, _ = checker.is_feasible(_workload(32, 1883))  # 60,256

    assert at_threshold is True
    assert just_over is False


def test_the_guard_reproduces_the_known_false_alarm() -> None:
    """The single false positive in the campaigns: 8 GPUs, per-device batch 32 x 2475 tokens
    = 79,200, which completed. The guard is expected to reject it; that is the known cost."""
    checker = TokenBudgetFeasibilityChecker(EMPIRICAL_OOM_TOKEN_BUDGET)

    feasible, metadata = checker.is_feasible(_workload(32, 2475))

    assert feasible is False
    assert metadata["tokens_per_device"] == 79_200


def test_the_guard_measures_per_device_load_not_effective_batch() -> None:
    """WorkloadSpec.batch_size is PER-DEVICE. Two configs with the same per-device load must get
    the same verdict however many GPUs they span, or the guard would track the wrong quantity."""
    checker = TokenBudgetFeasibilityChecker(EMPIRICAL_OOM_TOKEN_BUDGET)

    one_node = _workload(64, 2048)
    many_nodes = _workload(64, 2048)
    many_nodes.number_of_nodes = 16

    assert checker.is_feasible(one_node)[0] == checker.is_feasible(many_nodes)[0]
    assert checker.is_feasible(one_node)[1]["tokens_per_device"] == 64 * 2048


def test_a_nonsense_threshold_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        TokenBudgetFeasibilityChecker(0)


# --------------------------------------------------------------------------- #
# Composition: the guard may only veto
# --------------------------------------------------------------------------- #


def test_the_guard_never_promotes_an_infeasible_config() -> None:
    """It is a guard, not a second opinion: a backend `no` stays `no` however small the config."""
    backend = _Spy(verdict=False)
    guarded = GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(EMPIRICAL_OOM_TOKEN_BUDGET), backend)

    feasible, _ = guarded.is_feasible(_workload(1, 128))

    assert feasible is False


def test_a_vetoed_config_never_reaches_the_backend() -> None:
    """Guard-first is deliberate: the backend may be an AutoGluon model, and cheap arithmetic
    should not pay for a classifier call on a config it has already rejected."""
    backend = _Spy()
    guarded = GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(EMPIRICAL_OOM_TOKEN_BUDGET), backend)

    guarded.is_feasible(_workload(64, 4096))  # 262,144, far over

    assert backend.calls == 0


def test_a_passing_config_is_decided_by_the_backend_and_keeps_both_metadata() -> None:
    backend = _Spy()
    guarded = GuardedFeasibilityChecker(TokenBudgetFeasibilityChecker(EMPIRICAL_OOM_TOKEN_BUDGET), backend)

    feasible, metadata = guarded.is_feasible(_workload(8, 1024))

    assert feasible is True
    assert backend.calls == 1
    assert metadata["backend"] == "spy"
    assert metadata["guard"] == "token_budget"
    assert metadata["tokens_per_device"] == 8192


# --------------------------------------------------------------------------- #
# Wiring: opt-in, and it must not change the default
# --------------------------------------------------------------------------- #


def test_the_guard_is_off_by_default() -> None:
    """Enabling it by default would silently change every recommendation this project has made."""
    checker = create_feasibility_checker({"feasibility": "rules"})

    feasible, _ = checker.is_feasible(_workload(64, 4096))  # 262,144

    assert feasible is True


def test_the_guard_engages_when_the_config_asks_for_it() -> None:
    checker = create_feasibility_checker({"feasibility": "rules", "empirical_oom_guard": True})

    assert checker.is_feasible(_workload(64, 4096))[0] is False
    assert checker.is_feasible(_workload(8, 1024))[0] is True


def test_the_threshold_is_overridable() -> None:
    checker = create_feasibility_checker(
        {"feasibility": "rules", "empirical_oom_guard": True, "empirical_oom_token_budget": 8192}
    )

    assert checker.is_feasible(_workload(8, 1024))[0] is True  # exactly 8192, strict >
    assert checker.is_feasible(_workload(9, 1024))[0] is False


def test_the_guard_composes_with_every_backend_including_none() -> None:
    """`feasibility: none` means "no OOM model", not "no guard" — the guard is orthogonal."""
    checker = create_feasibility_checker({"feasibility": "none", "empirical_oom_guard": True})

    assert checker.is_feasible(_workload(64, 4096))[0] is False
    assert checker.is_feasible(_workload(8, 1024))[0] is True
