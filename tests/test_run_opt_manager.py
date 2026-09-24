import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_kernel_agent.kernel_backend import KernelBundle
from triton_kernel_agent.opt_manager import (
    OptimizationManager,
    _target_improvement_reached,
    _validate_target_improvement_pct,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "run_opt_manager.py"
_SPEC = importlib.util.spec_from_file_location("run_opt_manager", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_result = _MODULE.summarize_result
write_best_kernel = _MODULE.write_best_kernel


def _result(**overrides):
    result = {
        "success": True,
        "kernel_code": "def kernel_function(): pass\n",
        "best_time_ms": 0.8,
        "initial_kernel_time_ms": 1.0,
        "pytorch_baseline_ms": 1.2,
        "total_rounds": 1,
        "top_kernels": [{"time_ms": 0.8}],
    }
    result.update(overrides)
    return result


def test_summarize_result_distinguishes_outcomes():
    assert summarize_result(_result())["status"] == "SUCCESS"
    assert (
        summarize_result(_result(best_time_ms=1.0))["status"] == "NO_GAIN"
    )
    assert (
        summarize_result(_result(top_kernels=[]))["status"] == "DEGRADED"
    )
    assert (
        summarize_result(_result(success=False, top_kernels=[]))["status"]
        == "FAILED"
    )


def test_write_best_kernel_materializes_native_bundle(tmp_path):
    files = {
        "kernel.py": "def kernel_function(x): return x\n",
        "binding.cpp": "// binding\n",
        "kernel.mu": "// kernel\n",
        "setup.py": "# setup\n",
    }
    result = _result(kernel_code=KernelBundle(files).render_for_prompt())
    paths = write_best_kernel(result, "MUSA", tmp_path)

    serialized = tmp_path / "optimized_kernel_musa.py"
    assert paths["serialized_kernel"] == str(serialized)
    assert serialized.is_file()
    assert paths["best_bundle_dir"] == str(tmp_path / "best_bundle")
    for name, content in files.items():
        assert (tmp_path / "best_bundle" / name).read_text() == content


def test_optimization_summary_is_json_serializable():
    json.dumps(summarize_result(_result()))


def test_target_improvement_uses_initial_kernel_latency():
    assert not _target_improvement_reached(10.0, 8.1, 20.0)
    assert _target_improvement_reached(10.0, 8.0, 20.0)
    assert _target_improvement_reached(10.0, 7.5, 20.0)


def _run_target_scenario(tmp_path, target):
    best = {}

    def initialize(initial):
        best["value"] = initial

    def update(results, round_num):
        best["value"] = SimpleNamespace(
            program_id=f"round-{round_num}",
            kernel_code=f"kernel-{round_num}",
            metrics=SimpleNamespace(time_ms=results[0]["time_ms"]),
            generation=round_num,
        )

    strategy = SimpleNamespace(
        initialize=initialize,
        select_candidates=lambda round_num: [{"round": round_num}],
        update_with_results=update,
        get_best_program=lambda: best["value"],
    )
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.max_rounds = 3
    manager.strategy = strategy
    manager.experience_store = None
    manager.worker_kwargs = {}
    manager.logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    manager.database = SimpleNamespace(get_top_k=lambda _limit: [best["value"]])
    manager._baseline_profile_cache = {}
    manager._verify_initial_kernel = lambda *_args: True
    manager._benchmark_pytorch_baseline = lambda *_args: 12.0
    manager._benchmark_pytorch_compile = lambda *_args: 11.0
    manager._benchmark_initial_kernel = lambda *_args: 10.0
    manager._record_experiences = lambda _results: None

    rounds = []
    times = iter([9.0, 8.0, 7.0])

    def run_workers(_candidates, round_num, *_args):
        rounds.append(round_num)
        return [{"success": True, "worker_id": 0, "time_ms": next(times)}]

    manager._run_workers = run_workers
    problem = tmp_path / "problem.py"
    problem.write_text("class Model: pass\n")
    result = manager.run_optimization(
        initial_kernel="initial",
        problem_file=problem,
        test_code="test",
        max_rounds=3,
        target_improvement_pct=target,
    )
    return result, rounds


def test_target_stops_before_starting_the_next_round(tmp_path):
    result, rounds = _run_target_scenario(tmp_path, 20)
    assert rounds == [1, 2]
    assert result["total_rounds"] == 2
    assert result["target_reached"] is True
    assert result["termination_reason"] == "target_improvement_reached"


def test_blank_target_runs_every_round(tmp_path):
    result, rounds = _run_target_scenario(tmp_path, None)
    assert rounds == [1, 2, 3]
    assert result["target_reached"] is False


@pytest.mark.parametrize("value", [0, -1, 100.01, float("inf"), "20", True])
def test_target_improvement_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="target_improvement_pct"):
        _validate_target_improvement_pct(value)
    assert _validate_target_improvement_pct(None) is None
