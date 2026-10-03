"""The 'intelligent' cascade: an exact cache match, else a simulation predictor.

``CacheThenSimulatePredictor.predict`` (composite.py) does:
    if hit is not None and hit.predicted_throughput and hit.predicted_throughput > 0:
        return hit                        # a recorded run
    else:
        return fallback.predict(...)      # the simulation model

The tests build the cache and fallback predictions and check which object comes back.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coastline.sdk.predictors.performance.composite import CacheThenSimulatePredictor


class _Stub:
    """A predictor that returns a fixed (possibly None) prediction and counts calls."""

    def __init__(self, pred, name="stub"):
        self._pred = pred
        self._name = name
        self.calls = 0

    def predict(self, workload, context):
        self.calls += 1
        return self._pred

    def get_name(self):
        return self._name


class _Exploding:
    """Fallback predictor that raises if called."""

    def predict(self, workload, context):
        raise AssertionError("fallback was called despite a valid cache hit")

    def get_name(self):
        return "exploding"


def test_valid_cache_hit_is_returned_unchanged_and_short_circuits_fallback():
    # A hit with positive throughput (999) is returned as is, and the fallback is not called
    # (it would raise).
    hit = SimpleNamespace(predicted_throughput=999.0)
    p = CacheThenSimulatePredictor(cache=_Stub(hit), fallback=_Exploding())
    result = p.predict(None, None)
    assert result is hit


def test_cache_miss_none_falls_through_to_fallback_once():
    # A cache miss (None) returns the fallback's prediction from a single fallback call.
    fallback_pred = SimpleNamespace(predicted_throughput=111.0)
    fallback = _Stub(fallback_pred)
    p = CacheThenSimulatePredictor(cache=_Stub(None), fallback=fallback)
    result = p.predict(None, None)
    assert result is fallback_pred
    assert fallback.calls == 1


@pytest.mark.parametrize("bad_throughput", [0.0, -5.0, None])
def test_non_positive_or_missing_cache_throughput_is_treated_as_miss(bad_throughput):
    # A hit counts only with a truthy throughput above 0, so all three fall back:
    # 0.0 and None are falsy, and -5.0 is truthy but not above 0.
    hit = SimpleNamespace(predicted_throughput=bad_throughput)
    fallback_pred = SimpleNamespace(predicted_throughput=111.0)
    p = CacheThenSimulatePredictor(cache=_Stub(hit), fallback=_Stub(fallback_pred))
    result = p.predict(None, None)
    assert result is fallback_pred


def test_get_name_advertises_the_actual_fallback():
    # get_name includes the fallback's own get_name(), so a catboost fallback shows in run reports.
    p = CacheThenSimulatePredictor(cache=_Stub(None), fallback=_Stub(None, name="kavier"))
    assert p.get_name() == "intelligent (cache->kavier)"
    p2 = CacheThenSimulatePredictor(cache=_Stub(None), fallback=_Stub(None, name="catboost"))
    assert p2.get_name() == "intelligent (cache->catboost)"
