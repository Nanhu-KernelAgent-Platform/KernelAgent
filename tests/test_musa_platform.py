import logging

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
    problem = tmp_path / "problem.py"
    problem.write_text("pass\n")
    for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py"):
        (tmp_path / name).write_text("# test\n")

    worker_result = benchmarker.benchmark_kernel(tmp_path / "kernel.py", problem)
    assert worker_result == {"time_ms": 0.25, "speedup": 1.0, "ptx_hash": None}

    bundle = "".join(
        f"FILE: {name}\n```\n# test\n```\n"
        for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
    )
    assert benchmarker.benchmark_kernel(bundle, problem) == 0.25
    assert benchmarker.benchmark_pytorch(problem) == {"time_ms": 0.25}


def test_musa_benchmark_script_has_separate_backward_target():
    assert "KERNELAGENT_OPTIMIZATION_TARGET" in _BENCHMARK_SCRIPT
    assert "torch.autograd.grad" in _BENCHMARK_SCRIPT
    assert "retain_graph=True" in _BENCHMARK_SCRIPT
    assert _BENCHMARK_SCRIPT.index(
        "invoke = bind_kernel_call"
    ) < _BENCHMARK_SCRIPT.index("output = invoke()")


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
