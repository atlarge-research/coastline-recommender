"""The 'intelligent' cascade: exact cache match, else a simulation predictor.

Contract of ``CacheThenSimulatePredictor.predict`` (see composite.py):
    if hit is not None AND hit.predicted_throughput (truthy) AND > 0:
        return hit                        # a real recorded run wins
    else:
        return fallback.predict(...)      # fall back to the simulation model

Oracles below are the *routing decision* plus object identity: we construct
the cache/fallback predictions ourselves, so the expected object is an input we
control, never a value the code produced.
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
    """Fallback predictor that must never be consulted on a valid cache hit."""

    def predict(self, workload, context):
        raise AssertionError("fallback was called despite a valid cache hit")

    def get_name(self):
        return "exploding"


def test_valid_cache_hit_is_returned_verbatim_and_short_circuits_fallback():
    # Cache hit carries positive throughput (999 > 0), so the cascade MUST
    # return that exact object and MUST NOT fall through to the fallback (a cache
    # hit exists precisely to avoid a second, expensive simulation call).
    # Oracle: identity — the result is the very object the cache returned;
    # fallback.predict raising proves the short-circuit (it was not consulted).
    hit = SimpleNamespace(predicted_throughput=999.0)
    p = CacheThenSimulatePredictor(cache=_Stub(hit), fallback=_Exploding())
    result = p.predict(None, None)
    assert result is hit


def test_cache_miss_none_falls_through_to_fallback_once():
    # Cache returns None (no recorded run for this workload) -> the fallback
    # prediction is used verbatim. Oracle: the result is identically the
    # fallback object, and the fallback is consulted exactly once (no double call).
    fallback_pred = SimpleNamespace(predicted_throughput=111.0)
    fallback = _Stub(fallback_pred)
    p = CacheThenSimulatePredictor(cache=_Stub(None), fallback=fallback)
    result = p.predict(None, None)
    assert result is fallback_pred
    assert fallback.calls == 1


@pytest.mark.parametrize("bad_throughput", [0.0, -5.0, None])
def test_non_positive_or_missing_cache_throughput_is_treated_as_miss(bad_throughput):
    # The guard requires throughput to be truthy AND strictly > 0. Three kinds
    # of unusable cache entry must therefore fall through to the fallback:
    #   0.0  -> falsy, fails the truthiness clause
    #   -5.0 -> truthy but fails the `> 0` clause
    #   None -> falsy, fails the truthiness clause
    # Oracle: for every kind the fallback object (not the bad hit) is returned.
    hit = SimpleNamespace(predicted_throughput=bad_throughput)
    fallback_pred = SimpleNamespace(predicted_throughput=111.0)
    p = CacheThenSimulatePredictor(cache=_Stub(hit), fallback=_Stub(fallback_pred))
    result = p.predict(None, None)
    assert result is fallback_pred


def test_get_name_advertises_the_actual_fallback():
    # get_name reports the real fallback model, not a hardcoded "kavier" — so a config that
    # sets a different fallback (e.g. catboost) is visible in run reporting. Oracle: the label
    # embeds the fallback's own get_name().
    p = CacheThenSimulatePredictor(cache=_Stub(None), fallback=_Stub(None, name="kavier"))
    assert p.get_name() == "intelligent (cache→kavier)"
    p2 = CacheThenSimulatePredictor(cache=_Stub(None), fallback=_Stub(None, name="catboost"))
    assert p2.get_name() == "intelligent (cache→catboost)"
