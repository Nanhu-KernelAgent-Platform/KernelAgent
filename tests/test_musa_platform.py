import logging
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from triton_kernel_agent.platform.musa import (
    _BENCHMARK_SCRIPT,
    MusaBenchmarker,
    MusaProfilerMetadata,
    MusaProfilerResults,
    MusaKernelProfiler,
    MusaVerifier,
    parse_mcu_metrics,
)
from triton_kernel_agent.platform.registry import registry


@pytest.fixture
def mock_musa_runtime(monkeypatch):
    import torch
    synchronizations = []
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {'enable_timing': True}
        def record(self):
            pass
        def elapsed_time(self, other):
            return .4
    fake = ModuleType('torch')
    fake.__dict__.update(torch.__dict__)
    fake.manual_seed = lambda seed: None
    fake.empty_strided = lambda *args, **kwargs: torch.empty_strided(*args, **dict(kwargs, device='cpu'))
    fake.musa = SimpleNamespace(Event=Event, synchronize=lambda: synchronizations.append(True),
                               current_device=lambda: 0, get_device_name=lambda index: 'mock-device')
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setitem(sys.modules, 'torch_musa', ModuleType('torch_musa'))
    return synchronizations
from triton_kernel_agent.platform_config import get_platform


def test_musa_platform_configuration():
    config = get_platform("musa")
    assert config.device_string == "musa"
    assert config.torch_namespace == "musa"
    assert config.profiler_command == "mcu"


def test_musa_components_are_registered():
    for component in (
        "verifier",
        "benchmarker",
        "worker_runner",
        "specs_provider",
        "profiler",
        "roofline_analyzer",
        "bottleneck_analyzer",
        "rag_prescriber",
    ):
        assert registry.has(component, "musa")


def test_parse_mcu_metrics_normalizes_sol_fields():
    frame, metrics = parse_mcu_metrics(
        "Compute (MP) Throughput  73.5 %\nMemory Throughput  42.0 %\n"
    )
    assert len(frame) == 2
    assert metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 73.5
    assert (
        metrics["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]
        == 42.0
    )


def test_parse_mcu_13_table_with_unit_before_value():
    frame, metrics = parse_mcu_metrics(
        "Metric Name            Metric Unit Metric Value\n"
        "Compute(MP) Throughput           %         4.13\n"
        "Memory Throughput                %        10.15\n"
        "Elapsed Cycles               cycle       17,325\n"
    )
    assert len(frame) == 3
    assert metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 4.13
    assert (
        metrics["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]
        == 10.15
    )
    assert metrics["mcu::elapsed_cycles"] == 17325.0


def test_parse_mcu_keeps_percent_sol_when_absolute_throughput_reuses_name():
    frame, metrics = parse_mcu_metrics(
        "Metric Name            Metric Unit Metric Value\n"
        "Memory Throughput                %        45.32\n"
        "Memory Throughput          Gbyte/s       696.03\n"
    )
    assert len(frame) == 2
    assert (
        metrics["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"]
        == 45.32
    )
    assert metrics["mcu::memory_throughput"] == 45.32
    assert metrics["mcu::memory_throughput_gbyte_s"] == 696.03


def test_musa_benchmarker_supports_manager_and_worker_contracts(tmp_path, monkeypatch):
    class Lock:
        def acquire(self):
            return True

        def release(self):
            return None

    benchmarker = MusaBenchmarker(tmp_path, logging.getLogger(__name__), Lock())
    monkeypatch.setattr(benchmarker, "_run", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(benchmarker, "_run_measurement", lambda *args, **kwargs: {
        "time_ms": 0.25, "samples_ms": [0.24, 0.25, 0.26],
    })
    problem = tmp_path / "problem.py"
    problem.write_text("pass\n")
    for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py"):
        (tmp_path / name).write_text("# test\n")

    worker_result = benchmarker.benchmark_kernel(tmp_path / "kernel.py", problem)
    assert worker_result == {
        "time_ms": 0.25, "samples_ms": [0.24, 0.25, 0.26],
        "speedup": 1.0, "ptx_hash": None,
    }

    bundle = "".join(
        f"FILE: {name}\n```\n# test\n```\n"
        for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
    )
    assert benchmarker.benchmark_kernel(bundle, problem) == 0.25
    assert benchmarker.benchmark_pytorch(problem) == {"time_ms": 0.25}


@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_musa_benchmark_always_routes_through_case_suite(tmp_path, monkeypatch, capsys, direction, mock_musa_runtime):
    problem = tmp_path / "problem.py"
    problem.write_text(f"KERNELAGENT_OPTIMIZATION_TARGET = {direction!r}\n")
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel_function(x): return x\n")
    calls = []

    def benchmark_cases(reference, function, device, measured_direction, measure):
        assert reference.KERNELAGENT_OPTIMIZATION_TARGET == direction
        assert device == "musa" and measured_direction == direction
        assert callable(measure)
        calls.append(function)
        return {"time_ms": 0.25, "samples_ms": [0.24, 0.25, 0.26]}

    monkeypatch.setitem(sys.modules, "kernel_case_suite", SimpleNamespace(benchmark_cases=benchmark_cases, move=None))
    monkeypatch.setitem(sys.modules, "kernel_call_binding", SimpleNamespace(bind_kernel_call=None))
    # The reference intentionally has no Model: falling through to the old
    # primary-only path would fail rather than silently measure the wrong work.
    for mode in ("reference", "kernel"):
        monkeypatch.setattr(sys, "argv", ["benchmark.py", str(problem), mode, "1", "1", str(kernel)])
        exec(_BENCHMARK_SCRIPT, {})
        assert json.loads(capsys.readouterr().out)["hardware"] == "mock-device"
    assert calls[0] is None and callable(calls[1])


def test_musa_primary_reference_keeps_three_timing_windows(tmp_path, monkeypatch, capsys, mock_musa_runtime):
    problem = tmp_path / "problem.py"
    problem.write_text(
        "class Model:\n"
        "    def __init__(self): self.calls = 0\n"
        "    def to(self, device): return self\n"
        "    def __call__(self): self.calls += 1\n"
        "def get_inputs(): return []\n"
    )
    synchronizations = mock_musa_runtime
    invocations = []

    def benchmark_cases(reference, function, device, direction, measure):
        assert function is None and device == 'musa' and direction == 'forward'
        samples = measure(lambda: invocations.append(True))
        return {'time_ms': 0.1, 'samples_ms': samples}

    monkeypatch.setitem(sys.modules, "kernel_case_suite", SimpleNamespace(benchmark_cases=benchmark_cases))
    monkeypatch.setitem(sys.modules, "kernel_call_binding", SimpleNamespace(bind_kernel_call=None))
    monkeypatch.setattr(sys, "argv", ["benchmark.py", str(problem), "reference", "2", "4"])
    namespace = {}
    exec(_BENCHMARK_SCRIPT, namespace)
    result = json.loads(capsys.readouterr().out)
    assert result == {"time_ms": 0.1, "samples_ms": [0.1, 0.1, 0.1], "hardware": "mock-device"}
    assert len(invocations) == 2 + 3 * 4
    assert len(synchronizations) == 4


@pytest.mark.parametrize('mode', ['reference', 'kernel'])
def test_primary_benchmark_preserves_strides_without_extra_cases(tmp_path, monkeypatch, capsys, mode, mock_musa_runtime):
    from triton_kernel_agent.case_suite import benchmark_cases, verify_cases, measurement_is_covered
    from triton_kernel_agent.kernel_call_binding import bind_kernel_call
    problem = tmp_path / 'layout_problem.py'
    problem.write_text(
        'import torch\n'
        'class Model(torch.nn.Module):\n'
        '    def to(self, device): return self\n'
        '    def forward(self, x):\n'
        '        assert x.stride() == (14, 2)\n'
        '        assert x.dtype == torch.float32\n'
        '        return x * x\n'
        'def get_inputs(): return [torch.randn(5, 14)[:, ::2]]\n')
    kernel = tmp_path / 'layout_kernel.py'
    kernel.write_text(
        'def kernel_function(x):\n'
        '    assert x.stride() == (14, 2)\n'
        '    return x * x\n')
    monkeypatch.setitem(sys.modules, 'kernel_case_suite', SimpleNamespace(benchmark_cases=benchmark_cases))
    monkeypatch.setitem(sys.modules, 'kernel_call_binding', SimpleNamespace(bind_kernel_call=bind_kernel_call))
    monkeypatch.setattr(sys, 'argv', ['benchmark.py', str(problem), mode, '1', '2', str(kernel)])
    namespace = {}
    exec(_BENCHMARK_SCRIPT, namespace)
    result = json.loads(capsys.readouterr().out)
    assert result['samples_ms'] == [0.2, 0.2, 0.2]
    rows = result['case_measurements']
    assert len(rows) == 1 and rows[0]['case_key'] == 'default'
    assert rows[0]['inputs']['inputs'][0]['stride'] == [14, 2]
    assert rows[0]['samples_ms'] == result['samples_ms']
    checked = verify_cases(namespace['problem'], lambda x: x * x, 'cpu')
    assert measurement_is_covered(rows, checked)
    from triton_kernel_agent.opt_manager import OptimizationManager
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.benchmarker = SimpleNamespace(last_case_measurements=rows,
                                          benchmark_kernel_samples=lambda *args: result['samples_ms'])
    manager.verifier = SimpleNamespace(last_case_results=checked)
    manager.logger = logging.getLogger(__name__)
    assert manager._benchmark_initial_kernel_samples('', problem) == result['samples_ms']


def test_musa_sample_benchmark_builds_once_and_returns_three_windows(
    tmp_path, monkeypatch,
):
    class Lock:
        def acquire(self):
            return True

        def release(self):
            return None

    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if "setup.py" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({'time_ms': 0.1, 'samples_ms': [0.11, 0.09, 0.10],
                               'hardware': 'MTT S5000', 'case_measurements': [{
                                   'case_key': 'default', 'time_ms': 0.1,
                                   'samples_ms': [0.11, 0.09, 0.10], 'inputs': {}}]})+'\n',
            stderr="",
        )

    monkeypatch.setattr("triton_kernel_agent.platform.musa.subprocess.run", fake_run)
    benchmarker = MusaBenchmarker(tmp_path, logging.getLogger(__name__), Lock())
    bundle = "".join(
        f"FILE: {name}\n```\n# test\n```\n"
        for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
    )
    problem = tmp_path / "problem.py"
    problem.write_text("pass\n")
    assert benchmarker.benchmark_kernel_samples(bundle, problem) == [0.11, 0.09, 0.1]
    assert sum("setup.py" in cmd for cmd in calls) == 1
    assert benchmarker.last_case_measurements[0]['case_key'] == 'default'
    assert benchmarker.last_case_measurements[0]['samples_ms'] == [0.11, 0.09, 0.1]

    assert benchmarker.last_hardware == "MTT S5000"
    assert "torch.musa.get_device_name" in _BENCHMARK_SCRIPT

def test_musa_verifier_preserves_rendered_bundle(tmp_path, monkeypatch):
    captured = {}

    def fake_verify(self, kernel_code, test_code, problem_description, max_rounds=0):
        captured["files"] = sorted(kernel_code.files)
        return True, kernel_code, ""

    monkeypatch.setattr(
        "triton_kernel_agent.worker.VerificationWorker.verify_with_refinement",
        fake_verify,
    )
    problem = tmp_path / "problem.py"
    problem.write_text("pass\n")
    bundle = "".join(
        f"FILE: {name}\n```\n# test\n```\n"
        for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
    )
    verifier = MusaVerifier(tmp_path, logging.getLogger(__name__))
    assert verifier.verify(bundle, problem, ["pass"])
    assert captured["files"] == ["binding.cpp", "kernel.mu", "kernel.py", "setup.py"]
    assert (tmp_path / "initial_verify" / "problem.py").read_text() == "pass\n"


def test_mcu_cache_requires_exact_bundle_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSA_MCU_CACHE_DIR", str(tmp_path / "cache"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py"):
        (bundle / name).write_text(f"# {name}\n")
    kernel = bundle / "kernel.py"
    problem = tmp_path / "problem.py"
    problem.write_text("pass\n")
    frame, flat = parse_mcu_metrics("Compute (MP) Throughput  5.0 %\n")
    profiler = MusaKernelProfiler(log_dir=tmp_path)
    result = MusaProfilerResults(
        frame,
        {"kernel": flat},
        MusaProfilerMetadata(str(kernel), str(problem), 1, "now", "test"),
    )
    profiler._save_cached_profile(kernel, problem, result)
    assert profiler._load_cached_profile(kernel, problem, 2) is not None

    (bundle / "kernel.mu").write_text("# changed\n")
    assert profiler._load_cached_profile(kernel, problem, 2) is None

    (bundle / "kernel.mu").write_text("# kernel.mu\n")
    profiler._save_cached_profile(kernel, problem, result)
    problem.write_text("# changed problem shape\n")
    assert profiler._load_cached_profile(kernel, problem, 2) is None


def test_mcu_command_uses_default_sampling_and_preserves_reports(tmp_path):
    default_profiler = MusaKernelProfiler(log_dir=tmp_path)
    default_command = default_profiler._profile_command(
        tmp_path / "ncu_wrapper.py", tmp_path / "default.mcu-rep"
    )
    assert "--sampling-interval" not in default_command
    # MCU 1.1.1 rejects NCU's --launch-count option, sometimes with exit code 0.
    assert "--launch-count" not in default_command
    assert default_command[default_command.index("--output") + 1].endswith(
        "default.mcu-rep"
    )

    profiler = MusaKernelProfiler(log_dir=tmp_path, mcu_sampling_interval=1000)
    command = profiler._profile_command(
        tmp_path / "ncu_wrapper.py", tmp_path / "profile.mcu-rep"
    )
    assert "--dump-mode" not in command
    assert "--launch-count" not in command
    assert command[command.index("--sampling-interval") + 1] == "1000"
    assert command[command.index("--output") + 1].endswith("profile.mcu-rep")

    override = profiler._profile_command(
        tmp_path / "ncu_wrapper.py",
        tmp_path / "retry.mcu-rep",
        sampling_interval=900,
    )
    assert override[override.index("--sampling-interval") + 1] == "900"


def test_mcu_detects_terminal_pfm_collection_failures():
    assert MusaKernelProfiler._has_terminal_pfm_failure("==ERROR== pfm buffer overflow")
    assert MusaKernelProfiler._has_terminal_pfm_failure(
        "PFM dump overlap. Please increase sampling interval"
    )
    assert not MusaKernelProfiler._has_terminal_pfm_failure(
        "Metric Name  Metric Unit  Metric Value"
    )


def test_mcu_detects_zero_exit_command_line_failures():
    assert MusaKernelProfiler._has_terminal_command_failure(
        "==ERROR== unrecognised option '--launch-count'. Use --help for further details."
    )
    assert MusaKernelProfiler._has_terminal_command_failure(
        "error: unknown option --example"
    )
    assert not MusaKernelProfiler._has_terminal_command_failure(
        "Metric Name  Metric Unit  Metric Value"
    )
