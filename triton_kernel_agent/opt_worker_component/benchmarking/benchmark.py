# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified benchmarking for Triton kernels and PyTorch baselines.

This module consolidates kernel and PyTorch benchmarking with improved timing
utilities, L2 cache clearing, and comprehensive statistics.
"""

import json
import logging
import os
import math
import shutil
import subprocess
import sys
import statistics
import tempfile
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch

from triton_kernel_agent.opt_worker_component.searching.ptx_fingerprint import (
    ptx_hash_from_cache,
)

from triton_kernel_agent.opt_worker_component.benchmarking.timing import (
    compute_timing_stats,
    prepare_pytorch_model,
    time_with_cuda_events,
    time_with_triton_do_bench,
)


class BenchmarkLockManager:
    """Manages GPU benchmarking locks to prevent resource contention."""

    def __init__(self, lock: Any, worker_id: int, logger: logging.Logger):
        """Initialize the lock manager.

        Args:
            lock: Shared multiprocessing lock for serializing GPU access
            worker_id: Worker ID for logging
            logger: Logger instance
        """
        self.lock = lock
        self.worker_id = worker_id
        self.logger = logger

    def __enter__(self):
        """Acquire the benchmarking lock."""
        self.logger.info(f"⏳ Waiting for benchmark lock (worker {self.worker_id})...")
        self.lock.acquire()
        self.logger.info(f"🔓 Acquired benchmark lock (worker {self.worker_id})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the benchmarking lock."""
        try:
            self.lock.release()
            self.logger.info(f"🔒 Released benchmark lock (worker {self.worker_id})")
        except Exception as e:
            self.logger.warning(f"Failed to release benchmark lock: {e}")
        return False


class Benchmark:
    """Unified benchmark for Triton kernels and PyTorch baselines.

    Supports two modes:
    1. Subprocess mode: Runs benchmarks in isolated processes (for compatibility)
    2. Direct mode: Uses in-process timing utilities (faster, more flexible)
    """

    def __init__(
        self,
        logger: logging.Logger,
        artifacts_dir: Path,
        benchmark_lock: Any,
        worker_id: int = 0,
        warmup: int = 25,
        repeat: int = 100,
        timing_method: str = "cuda_event",
    ):
        """Initialize the benchmark.

        Args:
            logger: Logger instance
            artifacts_dir: Directory for benchmark artifacts
            benchmark_lock: Shared lock to serialize GPU benchmarking
            worker_id: Worker ID
            warmup: Number of warmup iterations (or warmup time in ms for do_bench)
            repeat: Number of repeat iterations (or rep time in ms for do_bench)
            timing_method: Timing method ("cuda_event", "do_bench", "host_time")
        """
        self.logger = logger
        self.artifacts_dir = artifacts_dir
        self.lock_manager = BenchmarkLockManager(benchmark_lock, worker_id, logger)
        self.warmup = warmup
        self.repeat = repeat
        self.timing_method = timing_method

    def benchmark_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        baseline_file: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Benchmark Triton kernel performance using subprocess isolation.

        Uses subprocess for crash protection of potentially buggy kernels.

        Args:
            kernel_file: Path to kernel file
            problem_file: Path to problem file
            baseline_file: Path to baseline kernel (optional)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - speedup: Speedup vs baseline
        """
        ptx_cache_dir = Path(tempfile.mkdtemp(prefix="triton_cache_bench_"))
        try:
            with (nullcontext() if getattr(self, "_paired_lock_held", False) else self.lock_manager):
                results_json = self.artifacts_dir / "benchmark_results.json"
                benchmark_script = Path(__file__).parent / "kernel_subprocess.py"

                # Use KERNEL_PROFILER_PYTHON (the PAR bootstrap) when set, like
                # ncu_profiler.py; bare sys.executable is un-bootstrapped in a PAR.
                bench_python = (
                    os.environ.get("KERNEL_PROFILER_PYTHON") or sys.executable
                )

                cmd = [
                    bench_python,
                    str(benchmark_script),
                    "--problem",
                    str(problem_file),
                    "--kernel",
                    str(kernel_file),
                    "--warmup",
                    str(self.warmup),
                    "--repeat",
                    str(self.repeat),
                    "--json",
                    str(results_json),
                    "--quiet",
                ]

                if baseline_file:
                    cmd.extend(["--baseline"])

                # Isolate this benchmark's Triton compilation cache so we can
                # capture its PTX for fingerprint-based dedup without being
                # contaminated by sibling workers' artifacts.
                env = {**os.environ, "TRITON_CACHE_DIR": str(ptx_cache_dir)}

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,
                )

                if result.returncode != 0:
                    error_msg = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "Unknown error"
                    )
                    self.logger.error(f"Kernel benchmark failed: {error_msg}")
                    return {"time_ms": float("inf"), "speedup": 0.0, "ptx_hash": None}

                with open(results_json, "r") as f:
                    results = json.load(f)

                kernel_name = kernel_file.stem
                kernel_results = results.get("kernels", {}).get(kernel_name, {})
                raw_samples = kernel_results.get("samples_ms", [])
                samples = (
                    [float(value) for value in raw_samples]
                    if isinstance(raw_samples, list) and all(
                        isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value > 0
                        for value in raw_samples
                    ) else []
                )

                # Capture the PTX fingerprint from the isolated cache dir.
                # A None result is graceful — dedup treats it as a singleton.
                ptx_hash = ptx_hash_from_cache(ptx_cache_dir)

                return {
                    "time_ms": (
                        statistics.median(samples) if samples
                        else kernel_results.get("time_ms", float("inf"))
                    ),
                    "samples_ms": samples,
                    "case_measurements": kernel_results.get("case_measurements", []),
                    "aggregation": kernel_results.get("aggregation", "single_case"),
                    "speedup": kernel_results.get("speedup", 1.0),
                    "ptx_hash": ptx_hash,
                }

        except Exception as e:
            self.logger.error(f"Kernel benchmark failed: {e}")
            return {"time_ms": float("inf"), "speedup": 0.0, "ptx_hash": None}
        finally:
            shutil.rmtree(ptx_cache_dir, ignore_errors=True)

    def benchmark_pair(self, parent, candidate, problem_file):
        from triton_kernel_agent.measurement_policy import interleaved_pair
        with self.lock_manager:
            self._paired_lock_held = True
            try:
                return interleaved_pair(lambda path: self.benchmark_kernel(path, problem_file), parent, candidate)
            finally:
                self._paired_lock_held = False

    def benchmark_pytorch(
        self,
        problem_file: Path,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, Any]:
        """Benchmark PyTorch baseline using direct in-process timing.

        Always uses direct mode (PyTorch is stable, doesn't need subprocess isolation).

        Args:
            problem_file: Path to problem file (must define Model class and get_inputs())
            dtype: Data type to use (default: auto-detect based on model parameters)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - stats: Full timing statistics (mean, std, min, max, all_times, etc.)
        """
        try:
            with self.lock_manager:
                from triton_kernel_agent.opt_worker_component.benchmarking.timing import import_module
                from triton_kernel_agent.case_suite import benchmark_cases
                reference = import_module(problem_file, "explicit_pytorch_reference")
                if (hasattr(reference, "get_test_cases") or
                        getattr(reference, "KERNELAGENT_OPTIMIZATION_TARGET", "forward") == "backward"):
                    def measure(invoke):
                        for _ in range(self.warmup):
                            invoke()
                        torch.cuda.synchronize()
                        samples = []
                        for _ in range(3):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(self.repeat):
                                invoke()
                            end.record()
                            torch.cuda.synchronize()
                            samples.append(start.elapsed_time(end) / self.repeat)
                        return samples
                    return benchmark_cases(reference, None, "cuda",
                        getattr(reference, "KERNELAGENT_OPTIMIZATION_TARGET", "forward"), measure)
                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        lambda: model(*inputs),
                        [],
                        warmup=self.warmup,
                        rep=self.repeat,
                        verbose=False,
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        lambda: model(*inputs),
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)

                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                }

        except Exception as e:
            self.logger.error(f"PyTorch baseline benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf")}

    def benchmark_pytorch_compile(
        self,
        problem_file: Path,
        dtype: Optional[torch.dtype] = None,
    ) -> dict[str, Any]:
        """Benchmark torch.compile'd PyTorch baseline using direct in-process timing.

        Mirrors benchmark_pytorch() but wraps the model with torch.compile()
        and uses extended warmup (3 forward calls) before timing to allow
        compilation and warm caches.

        Args:
            problem_file: Path to problem file (must define Model class and get_inputs())
            dtype: Data type to use (default: auto-detect based on model parameters)

        Returns:
            Dictionary with benchmark results:
                - time_ms: Mean time in ms
                - stats: Full timing statistics (mean, std, min, max, all_times, etc.)
        """
        try:
            with self.lock_manager:
                from triton_kernel_agent.opt_worker_component.benchmarking.timing import import_module
                reference = import_module(problem_file, "explicit_compiled_reference")
                if (hasattr(reference, "get_test_cases") or
                        getattr(reference, "KERNELAGENT_OPTIMIZATION_TARGET", "forward") == "backward"):
                    self.logger.info("Compiled baseline skipped: primary-only timing cannot represent an explicit suite")
                    return {"time_ms": float("inf"), "reason": "explicit_suite_compile_baseline_not_supported"}
                model, inputs = prepare_pytorch_model(
                    problem_file=problem_file,
                    device="cuda",
                    dtype=dtype,
                )

                model = torch.compile(model)

                # Extended warmup: 3 forward calls to trigger compilation
                for _ in range(3):
                    model(*inputs)
                torch.cuda.synchronize()

                if self.timing_method == "do_bench":
                    times = time_with_triton_do_bench(
                        lambda: model(*inputs),
                        [],
                        warmup=self.warmup,
                        rep=self.repeat,
                        verbose=False,
                    )
                else:  # cuda_event
                    times = time_with_cuda_events(
                        lambda: model(*inputs),
                        [],
                        num_warmup=self.warmup,
                        num_trials=self.repeat,
                        clear_cache=True,
                        verbose=False,
                    )

                stats = compute_timing_stats(times)

                return {
                    "time_ms": stats["mean"],
                    "stats": stats,
                }

        except Exception as e:
            self.logger.error(f"PyTorch compile benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            return {"time_ms": float("inf")}
