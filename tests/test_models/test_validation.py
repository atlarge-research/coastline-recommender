"""Unit tests for ``coastline.sdk.models.validation``.

These exercise the shared workload-validation helpers used by the predictors:
``estimate_memory_requirement``, ``validate_gpu_memory`` and the
``validate_workload`` aggregate. The memory model is intentionally simple
(``0.5 * batch_size * tokens/1024 * 3``). Every expected value below is
hand-computed from first principles and written as a literal (a *different*
form than the implementation), never recomputed from the same formula.

Fast, deterministic, no I/O.
"""

import pytest

from coastline.sdk.exceptions import UnsupportedGPUError
from coastline.sdk.library.hardware import GPU_SPECS, get_gpu_memory
from coastline.sdk.models.validation import (
    estimate_memory_requirement,
    validate_gpu_memory,
    validate_workload,
)
from coastline.sdk.models.workload import WorkloadSpec


def _workload(gpu_model="A100-SXM4-40GB", tokens_per_sample=1024, batch_size=2, **kw):
    return WorkloadSpec(
        llm_model=kw.pop("llm_model", "gpt2"),
        fine_tuning_method=kw.pop("fine_tuning_method", "full"),
        gpu_model=gpu_model,
        tokens_per_sample=tokens_per_sample,
        batch_size=batch_size,
        **kw,
    )


# --------------------------------------------------------------------------
# estimate_memory_requirement
# --------------------------------------------------------------------------


class TestEstimateMemoryRequirement:
    @pytest.mark.parametrize(
        "batch,tokens,expected",
        [
            # Oracles hand-computed as 0.5 * batch * (tokens/1024) * 3, written as
            # literals so the test is independent of the impl's expression:
            (1, 1024, 1.5),  # 0.5 * 1 * 1    * 3 = 1.5  -> pins the 0.5*3 coefficient
            (2, 1024, 3.0),  # 0.5 * 2 * 1    * 3 = 3.0  -> linear in batch
            (1, 2048, 3.0),  # 0.5 * 1 * 2    * 3 = 3.0  -> linear in sequence length
            (1, 512, 0.75),  # 0.5 * 1 * 0.5  * 3 = 0.75 -> sub-1024 normalization
            (128, 4096, 768.0),  # 0.5 * 128 * 4 * 3 = 768.0 -> product of both terms
        ],
    )
    def test_matches_hand_computed_values(self, batch, tokens, expected):
        w = _workload(tokens_per_sample=tokens, batch_size=batch)
        assert estimate_memory_requirement(w) == pytest.approx(expected)

    def test_at_1024_tokens_equals_1_5_times_batch(self):
        # tokens/1024 == 1, so 0.5 * batch * 1 * 3 == 1.5 * batch; batch=8 -> 12.0.
        w = _workload(tokens_per_sample=1024, batch_size=8)
        assert estimate_memory_requirement(w) == pytest.approx(12.0)

    def test_scales_linearly_with_batch_size(self):
        # Scaling law: memory is linear in batch_size, so 32x the batch => exactly
        # 32x the memory. (A monotone-only check would pass on a wrong exponent.)
        small = estimate_memory_requirement(_workload(batch_size=2))
        big = estimate_memory_requirement(_workload(batch_size=64))  # 64/2 = 32x
        assert big == pytest.approx(32.0 * small)

    def test_scales_linearly_with_sequence_length(self):
        # Scaling law: memory is linear in tokens_per_sample, so 16x tokens => 16x mem.
        short = estimate_memory_requirement(_workload(tokens_per_sample=512))
        long = estimate_memory_requirement(_workload(tokens_per_sample=8192))  # 8192/512 = 16x
        assert long == pytest.approx(16.0 * short)

    def test_independent_of_gpu_model(self):
        # The estimate depends only on the workload shape, not the device: both use
        # batch=2, tokens=1024 -> 0.5*2*1*3 = 3.0 GB regardless of the GPU picked.
        a = estimate_memory_requirement(_workload(gpu_model="A100-SXM4-40GB"))
        b = estimate_memory_requirement(_workload(gpu_model="A100-SXM4-80GB"))
        assert a == pytest.approx(3.0)
        assert a == pytest.approx(b)


# --------------------------------------------------------------------------
# validate_gpu_memory
# --------------------------------------------------------------------------


class TestValidateGpuMemory:
    def test_small_workload_fits(self):
        # required = 0.5 * 2 * 1 * 3 = 3.0 GB, well under the 40 GB card.
        w = _workload(gpu_model="A100-SXM4-40GB", tokens_per_sample=1024, batch_size=2)
        ok, err = validate_gpu_memory(w)
        assert ok is True
        assert err == ""

    def test_oversized_workload_rejected(self):
        # required = 0.5 * 128 * (4096/1024) * 3 = 768 GB >> 40 GB.
        w = _workload(gpu_model="A100-SXM4-40GB", tokens_per_sample=4096, batch_size=128)
        ok, err = validate_gpu_memory(w)
        assert ok is False
        assert "exceeds" in err
        assert "A100-SXM4-40GB" in err
        assert "768.0GB" in err  # the derived requirement is surfaced in the message

    def test_larger_gpu_accommodates_more(self):
        # required = 0.5 * 40 * (1024/1024) * 3 = 60 GB, which sits between the
        # 40 GB and 80 GB A100 memory sizes: rejected on the 40, accepted on the 80.
        w = _workload(tokens_per_sample=1024, batch_size=40)
        assert estimate_memory_requirement(w) == pytest.approx(60.0)
        assert (
            validate_gpu_memory(_workload(gpu_model="A100-SXM4-40GB", tokens_per_sample=1024, batch_size=40))[0]
            is False
        )
        assert (
            validate_gpu_memory(_workload(gpu_model="A100-SXM4-80GB", tokens_per_sample=1024, batch_size=40))[0] is True
        )

    def test_unknown_gpu_raises_unsupported(self):
        # Unknown GPUs must fail loudly, not fabricate a 40GB default.
        with pytest.raises(UnsupportedGPUError):
            get_gpu_memory("DOES-NOT-EXIST")

    def test_unknown_gpu_makes_workload_invalid(self):
        # validate_gpu_memory must surface an unknown GPU as invalid rather
        # than validating against a fabricated memory size.
        w = _workload(gpu_model="DOES-NOT-EXIST", tokens_per_sample=1024, batch_size=2)
        ok, err = validate_gpu_memory(w)
        assert ok is False
        assert "DOES-NOT-EXIST" in err

    def test_required_exactly_equal_to_capacity_is_accepted(self):
        # The check is a strict '>', so required == available must still fit.
        # L40S has 48 GB. required = 0.5 * 32 * (1024/1024) * 3 = 48.0 == capacity.
        gpu = "L40S"
        assert GPU_SPECS[gpu]["memory_gb"] == 48
        at_cap = _workload(gpu_model=gpu, tokens_per_sample=1024, batch_size=32)
        assert estimate_memory_requirement(at_cap) == pytest.approx(48.0)
        assert validate_gpu_memory(at_cap)[0] is True

    def test_required_one_step_over_capacity_is_rejected(self):
        # L40S = 48 GB. required = 0.5 * 33 * 1 * 3 = 49.5 GB > 48 -> rejected.
        over = _workload(gpu_model="L40S", tokens_per_sample=1024, batch_size=33)
        assert estimate_memory_requirement(over) == pytest.approx(49.5)
        assert validate_gpu_memory(over)[0] is False


# --------------------------------------------------------------------------
# validate_workload (aggregate)
# --------------------------------------------------------------------------


class TestValidateWorkload:
    def test_valid_workload_passes(self):
        # required = 0.5 * 4 * 1 * 3 = 6.0 GB, well under the 80 GB card.
        w = _workload(gpu_model="A100-SXM4-80GB", tokens_per_sample=1024, batch_size=4)
        ok, err = validate_workload(w)
        assert ok is True
        assert err == ""

    def test_memory_failure_propagates(self):
        # required = 0.5 * 128 * (4096/1024) * 3 = 768 GB >> 40 GB; the memory
        # error message from validate_gpu_memory must be surfaced verbatim.
        w = _workload(gpu_model="A100-SXM4-40GB", tokens_per_sample=4096, batch_size=128)
        ok, err = validate_workload(w)
        assert ok is False
        assert "exceeds" in err
        assert "GPU memory" in err
