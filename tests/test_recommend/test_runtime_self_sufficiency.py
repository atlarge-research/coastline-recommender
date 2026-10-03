"""The predictors find their own files and load one ML runtime at a time.

The data-driven predictors work without ``trainer``, ``kavier/src`` or ``DATA_DIR`` being supplied,
and importing one ML predictor does not import the others: the native OpenMP runtimes of torch,
xgboost and catboost in one process segfault on macOS, so the playground runs each model on its
own. The web interface depends on both.

Each check runs in a subprocess with a minimal PYTHONPATH and asserts invariants (finite output,
GPU-count scaling, datasheet power bounds) that a stub returning a constant would fail.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


# Only the roots a bare `uvicorn coastline.ui.app:app` needs. trainer, kavier/src and
# trace-archive must be found by the code itself.
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


# Subprocess preamble: a workload factory that varies only the GPUs per node, so a 1-GPU and an
# 8-GPU run differ only in the multi-GPU scaling.
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
    """Kavier finds kavier/src under a minimal environment, its throughput rises sub-linearly
    with GPU count, and its per-GPU power lies in the A100 datasheet range."""
    code = _PREAMBLE + (
        "kp=PolicyFactory.throughput_predictor({'performance':'kavier'})\n"
        "p1=kp.predict(wl(1),CTX)\n"
        "p8=kp.predict(wl(8),CTX)\n"
        "assert p1 is not None and p8 is not None, 'kavier self-location failed (None prediction)'\n"
        "t1,t8=p1.predicted_throughput,p8.predicted_throughput\n"
        # finite, positive tokens/s
        "assert t1 and t8 and math.isfinite(t1) and math.isfinite(t8) and t1>0 and t8>0, (t1,t8)\n"
        # 8 GPUs are faster than 1
        "assert t8 > t1, f'8 GPUs ({t8}) not faster than 1 ({t1})'\n"
        # but less than 8x faster, since collective communication grows with the GPU count
        "assert t8 < 8*t1, f'scaling is not sub-linear: t8={t8} >= 8*t1={8*t1}'\n"
        # A100-SXM4-80GB datasheet: idle 75 W, TDP 400 W (GPU_SPECS).
        "assert p8.predicted_power is not None and 75.0 <= p8.predicted_power <= 400.0, "
        "f'per-GPU power {p8.predicted_power} outside [75,400]W'\n"
        # per-GPU power does not depend on the GPU count
        "assert p1.predicted_power == p8.predicted_power, (p1.predicted_power, p8.predicted_power)\n"
        "print('OK')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


@pytest.mark.lfs_model("random_forest")
def test_random_forest_self_locates_trainer_and_scales_sublinearly_with_minimal_env():
    """random_forest finds trainer and trace-archive without DATA_DIR and predicts a finite
    throughput that rises sub-linearly with GPU count."""
    code = _PREAMBLE + (
        "rf=PolicyFactory.throughput_predictor({'performance':'random_forest'})\n"
        "p1=rf.predict(wl(1),CTX)\n"
        "p8=rf.predict(wl(8),CTX)\n"
        "assert p1 is not None and p8 is not None, 'random_forest self-location failed (None prediction)'\n"
        "t1,t8=p1.predicted_throughput,p8.predicted_throughput\n"
        # finite, positive tokens/s for a known workload
        "assert t1 and t8 and math.isfinite(t1) and math.isfinite(t8) and t1>0 and t8>0, (t1,t8)\n"
        # faster on 8 GPUs, but less than 8x
        "assert t8 > t1, f'8 GPUs ({t8}) not faster than 1 ({t1})'\n"
        "assert t8 < 8*t1, f'scaling is not sub-linear: t8={t8} >= 8*t1={8*t1}'\n"
        # an unknown model gives no prediction
        "unknown=WorkloadSpec(llm_model='not-a-real-model',fine_tuning_method='full',gpu_model=G,"
        "tokens_per_sample=2048,batch_size=8,gpus_per_node=8,number_of_nodes=1)\n"
        "assert rf.predict(unknown,CTX) is None, 'out-of-library workload was not rejected'\n"
        "print('OK')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout


def test_importing_one_predictor_does_not_load_torch():
    """Importing the sklearn portfolio predictor does not import torch, so the playground
    subprocess for a model such as xgboost loads only its own backend."""
    code = (
        "import sys\n"
        "import importlib\n"
        "mod=importlib.import_module('coastline.sdk.predictors.performance.data_driven.sklearn_portfolio')\n"
        # the real module was imported
        "assert hasattr(mod, 'SklearnPortfolioPredictor'), 'sklearn_portfolio missing its predictor'\n"
        # and torch was not
        "assert 'torch' not in sys.modules, 'torch was imported transitively (eager __init__ regression)'\n"
        "print('OK')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout
