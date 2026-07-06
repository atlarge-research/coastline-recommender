"""The 'intelligent' cascade: exact cache match, else Kavier physics.

Contract of ``CacheThenPhysicsPredictor.predict`` (see composite.py):
    if hit is not None AND hit.predicted_throughput (truthy) AND > 0:
        return hit                       # a real recorded run wins
    else:
        return physics.predict(...)      # fall back to the analytical engine

Oracles below are the *routing decision* plus object identity: we construct
the cache/physics predictions ourselves, so the expected object is an input we
control, never a value the code produced.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coastline.sdk.predictors.performance.composite import CacheThenPhysicsPredictor


class _Stub:
    """A predictor that returns a fixed (possibly None) prediction and counts calls."""

    def __init__(self, pred):
        self._pred = pred
        self.calls = 0

    def predict(self, workload, context):
        self.calls += 1
        return self._pred

    def get_name(self):
        return "stub"


class _Exploding:
    """Physics predictor that must never be consulted on a valid cache hit."""

    def predict(self, workload, context):
        raise AssertionError("physics was called despite a valid cache hit")

    def get_name(self):
        return "exploding"


def test_valid_cache_hit_is_returned_verbatim_and_short_circuits_physics():
    # Cache hit carries positive throughput (999 > 0), so the cascade MUST
    # return that exact object and MUST NOT fall through to physics (a cache
    # hit exists precisely to avoid a second, expensive engine call).
    # Oracle: identity — the result is the very object the cache returned;
    # physics.predict raising proves the short-circuit (it was not consulted).
    hit = SimpleNamespace(predicted_throughput=999.0)
    p = CacheThenPhysicsPredictor(cache=_Stub(hit), physics=_Exploding())
    result = p.predict(None, None)
    assert result is hit


def test_cache_miss_none_falls_through_to_physics_once():
    # Cache returns None (no recorded run for this workload) -> the physics
    # prediction is used verbatim. Oracle: the result is identically the
    # physics object, and physics is consulted exactly once (no double call).
    physics_pred = SimpleNamespace(predicted_throughput=111.0)
    physics = _Stub(physics_pred)
    p = CacheThenPhysicsPredictor(cache=_Stub(None), physics=physics)
    result = p.predict(None, None)
    assert result is physics_pred
    assert physics.calls == 1


@pytest.mark.parametrize("bad_throughput", [0.0, -5.0, None])
def test_non_positive_or_missing_cache_throughput_is_treated_as_miss(bad_throughput):
    # The guard requires throughput to be truthy AND strictly > 0. Three kinds
    # of unusable cache entry must therefore fall through to physics:
    #   0.0  -> falsy, fails the truthiness clause
    #   -5.0 -> truthy but fails the `> 0` clause
    #   None -> falsy, fails the truthiness clause
    # Oracle: for every kind the physics object (not the bad hit) is returned.
    hit = SimpleNamespace(predicted_throughput=bad_throughput)
    physics_pred = SimpleNamespace(predicted_throughput=111.0)
    p = CacheThenPhysicsPredictor(cache=_Stub(hit), physics=_Stub(physics_pred))
    result = p.predict(None, None)
    assert result is physics_pred


def test_get_name_advertises_the_cache_then_kavier_cascade():
    # Contract string consumed by the predictor resolver / run reporting; the
    # class spec is "exact cache match first, else Kavier physics". Pinned to
    # the exact documented label so a rename that would confuse the resolver
    # is caught.
    p = CacheThenPhysicsPredictor(cache=_Stub(None), physics=_Stub(None))
    assert p.get_name() == "intelligent (cache→kavier)"
