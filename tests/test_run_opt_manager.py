import json
import importlib.util
from pathlib import Path

from triton_kernel_agent.kernel_backend import KernelBundle

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
