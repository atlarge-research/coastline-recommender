"""Runtime self-sufficiency of the predictors.

These guard regressions that broke the web interface: the data-driven predictors
must work without `trainer`, `kavier/src` or `DATA_DIR` being supplied externally
(the code self-locates them), and importing one ML predictor must not drag every
ML runtime into the process — coexisting native OpenMP runtimes (torch + xgboost +
catboost) segfault on macOS, which is why the playground isolates each model.

Each subprocess runs under a DELIBERATELY MINIMAL PYTHONPATH (only the umbrella +
common roots — no trainer, kavier/src or DATA_DIR). A predictor that builds AND
produces physically sane output there has self-located every artifact it needs.
The oracles below are engine-independent invariants (finiteness, GPU-count scaling
law, datasheet power bounds), never a pinned model output — so they verify the
self-located engine actually computes, not merely that it returned some constant.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _model_is_materialized(stem: str) -> bool:
    """True if the resolved models pickle is a real artifact, not an unpulled Git-LFS pointer
    (CI checks out without LFS, where the file is a small `version https://git-lfs...` text stub).
    Uses the production resolver so the custom/ > packaged portfolio split is honoured."""
    from coastline.sdk.predictors.performance.data_driven.ml_common import performance_trained_model_path

    path = performance_trained_model_path(stem)
    return path.exists() and not path.read_bytes()[:40].startswith(b"version https://git-lfs")


# Deliberately MINIMAL: only the roots a bare `uvicorn coastline.ui.app:app` needs.
# trainer + kavier/src + trace-archive must be discovered by the code itself.
# Both the `api` and `recommender` packages live directly under the umbrella
# `coastline/`; `coastline_common` lives under `coastline/common`.
_UMBRELLA = REPO_ROOT / "coastline"
MINIMAL_PYTHONPATH = f"{_UMBRELLA}:{_UMBRELLA / 'common'}"


def _run(code: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {
        "PYTHONPATH": MINIMAL_PYTHONPATH,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)


# Shared subprocess preamble: build the predictor + a workload/context factory that
# only varies the GPU *layout* (per-node count), holding batch/tokens/GPU fixed so a
# 1-GPU vs 8-GPU pair isolates the multi-GPU scaling term.
_PREAMBLE = (
    "import math\n"
    "from coastline.sdk.policies import PolicyFactory\n"
    "from coastline.sdk.models.workload import WorkloadSpec\n"
    "from coastline.sdk.models.context import Constraints, SystemContext\n"
    "from coastline.sdk.library.hardware import get_gpu_memory\n"
    "G='NVIDIA-A100-SXM4-80GB'\n"
    "def wl(gpus_per_node):\n"
    "    return WorkloadSpec(llm_model='mistral-7b-v0.1',fine_tuning_method='full',gpu_model=G,"
    "tokens_per_sample=2048,batch_size=8,gpus_per_node=gpus_per_node,number_of_nodes=1)\n"
    "CTX=SystemContext(available_gpu_models=[G],max_gpus=8,gpu_memory={G:get_gpu_memory(G)},"
    "constraints=Constraints(max_gpus=8,gpus_per_node=8,max_nodes=1))\n"
)


def test_kavier_self_locates_and_scales_sublinearly_with_minimal_env():
    """Kavier self-locates kavier/src under a minimal env AND the built physics
    engine actually computes: total throughput rises with GPU count but strictly
    sub-linearly (communication overhead), and per-GPU power sits in the A100
    datasheet band. These invariants reject a stub that returns a constant."""
    code = _PREAMBLE + (
        "kp=PolicyFactory.throughput_predictor({'performance':'kavier'})\n"
        "p1=kp.predict(wl(1),CTX)\n"
        "p8=kp.predict(wl(8),CTX)\n"
        "assert p1 is not None and p8 is not None, 'kavier self-location failed (None prediction)'\n"
        "t1,t8=p1.predicted_throughput,p8.predicted_throughput\n"
        # Invariant: a real analytic engine yields finite, positive tokens/sec.
        "assert t1 and t8 and math.isfinite(t1) and math.isfinite(t8) and t1>0 and t8>0, (t1,t8)\n"
        # Scaling law: 8 GPUs must be FASTER than 1 (kills a constant-return stub)...
        "assert t8 > t1, f'8 GPUs ({t8}) not faster than 1 ({t1})'\n"
        # ...but STRICTLY sub-linear: <8x because collective-comm overhead grows with
        # GPU count (kills a linear total = per_gpu*count bug, which would give ==8x).
        "assert t8 < 8*t1, f'scaling is not sub-linear: t8={t8} >= 8*t1={8*t1}'\n"
        # Per-GPU power oracle from the NVIDIA A100-SXM4-80GB datasheet: idle 75W,
        # TDP 400W (see GPU_SPECS). Any calibrated draw must fall in [idle, TDP].
        "assert p8.predicted_power is not None and 75.0 <= p8.predicted_power <= 400.0, "
        "f'per-GPU power {p8.predicted_power} outside [75,400]W'\n"
        # Per-GPU power is intensive: it must not scale with the GPU *count*.
        "assert p1.predicted_power == p8.predicted_power, (p1.predicted_power, p8.predicted_power)\n"
        "print('OK')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


@pytest.mark.skipif(
    not _model_is_materialized("random_forest"),
    reason="random_forest model is an unpulled Git-LFS pointer (run `git lfs pull`)",
)
def test_random_forest_self_locates_trainer_and_scales_sublinearly_with_minimal_env():
    """A sklearn predictor self-locates trainer + trace-archive (no DATA_DIR) and
    produces a working model: finite positive throughput that rises sub-linearly
    with GPU count. ML output value is model-specific, so we assert only invariants,
    never a pinned number."""
    code = _PREAMBLE + (
        "rf=PolicyFactory.throughput_predictor({'performance':'random_forest'})\n"
        "p1=rf.predict(wl(1),CTX)\n"
        "p8=rf.predict(wl(8),CTX)\n"
        "assert p1 is not None and p8 is not None, 'random_forest self-location failed (None prediction)'\n"
        "t1,t8=p1.predicted_throughput,p8.predicted_throughput\n"
        # Invariant: finite, positive tokens/sec for an in-library workload.
        "assert t1 and t8 and math.isfinite(t1) and math.isfinite(t8) and t1>0 and t8>0, (t1,t8)\n"
        # Scaling law: more GPUs => higher total throughput, but sub-linear (<8x).
        "assert t8 > t1, f'8 GPUs ({t8}) not faster than 1 ({t1})'\n"
        "assert t8 < 8*t1, f'scaling is not sub-linear: t8={t8} >= 8*t1={8*t1}'\n"
        # Contract: an out-of-library model must not be silently extrapolated.
        "unknown=WorkloadSpec(llm_model='not-a-real-model',fine_tuning_method='full',gpu_model=G,"
        "tokens_per_sample=2048,batch_size=8,gpus_per_node=8,number_of_nodes=1)\n"
        "assert rf.predict(unknown,CTX) is None, 'out-of-library workload was not rejected'\n"
        "print('OK')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


def test_importing_one_predictor_does_not_load_torch():
    """Lazy package init: importing a non-torch predictor must not import torch, so the
    playground subprocess for e.g. xgboost (a sklearn-portfolio model) loads its backend
    alone (no segfault). Falsification: an eager `__init__` that imports torch turns the
    invariant red."""
    code = (
        "import sys\n"
        "import importlib\n"
        "mod=importlib.import_module('coastline.sdk.predictors.performance.data_driven.sklearn_portfolio')\n"
        # Positive control: prove we imported the real module (not a silent no-op that
        # would make the torch check trivially pass) — it must expose its predictor.
        "assert hasattr(mod, 'SklearnPortfolioPredictor'), 'sklearn_portfolio missing its predictor'\n"
        # The property under test: the non-torch predictor pulled in zero torch runtime.
        "assert 'torch' not in sys.modules, 'torch was imported transitively (eager __init__ regression)'\n"
        "print('OK')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout
