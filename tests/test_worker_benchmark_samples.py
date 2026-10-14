import json
import logging
from types import SimpleNamespace

from triton_kernel_agent.kernel_backend import KernelBundle
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from triton_kernel_agent.opt_worker_component.benchmarking import benchmark as module


class _Lock:
    def acquire(self):
        pass

    def release(self):
        pass


def test_worker_benchmark_preserves_timing_windows(tmp_path, monkeypatch):
    def fake_run(cmd, **_kwargs):
        output = cmd[cmd.index("--json") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            json.dump({"kernels": {"candidate": {
                "time_ms": 9.0, "samples_ms": [0.12, 0.10, 0.11],
            }}}, stream)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "ptx_hash_from_cache", lambda _path: None)
    benchmarker = module.Benchmark(
        logger=logging.getLogger(__name__), artifacts_dir=tmp_path,
        benchmark_lock=_Lock(),
    )
    result = benchmarker.benchmark_kernel(
        tmp_path / "candidate.py", tmp_path / "problem.py"
    )
    assert result["time_ms"] == 0.11
    assert result["samples_ms"] == [0.12, 0.10, 0.11]


def test_worker_benchmark_accepts_legacy_single_time(tmp_path, monkeypatch):
    def fake_run(cmd, **_kwargs):
        output = cmd[cmd.index("--json") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            json.dump({"kernels": {"candidate": {"time_ms": 0.25}}}, stream)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "ptx_hash_from_cache", lambda _path: None)
    benchmarker = module.Benchmark(
        logger=logging.getLogger(__name__), artifacts_dir=tmp_path,
        benchmark_lock=_Lock(),
    )
    result = benchmarker.benchmark_kernel(
        tmp_path / "candidate.py", tmp_path / "problem.py"
    )
    assert result["time_ms"] == 0.25
    assert result["samples_ms"] == []


def test_paired_native_bundles_keep_parent_and_candidate_separate(tmp_path):
    sources = []

    def fake_benchmark(kernel_file, _problem_file):
        source = kernel_file.read_text(encoding="utf-8")
        sources.append((kernel_file, source))
        latency = 0.2 if "parent" in source else 0.1
        return {"time_ms": latency, "samples_ms": [latency] * 3}

    def bundle(marker):
        return KernelBundle({
            "kernel.py": f"# {marker}\n",
            "binding.cpp": f"// {marker}\n",
            "kernel.mu": f"// {marker}\n",
            "setup.py": f"# {marker}\n",
        }).render_for_prompt()

    worker = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    worker.artifact_dir = tmp_path
    worker.logger = logging.getLogger(__name__)
    worker.benchmarker = SimpleNamespace(benchmark_kernel=fake_benchmark)
    parent, candidate = worker._benchmark_paired_candidate(
        bundle("parent"), bundle("candidate"), tmp_path / "problem.py", 1,
    )
    assert parent["time_ms"] == 0.2
    assert candidate["time_ms"] == 0.1
    assert [source for _, source in sources] == ["# parent\n", "# candidate\n"]
    assert sources[0][0].parent != sources[1][0].parent
    assert (sources[0][0].parent / "kernel.mu").read_text() == "// parent\n"
    assert (sources[1][0].parent / "kernel.mu").read_text() == "// candidate\n"
