"""MUSA platform components.

The implementation is intentionally isolated from the NVIDIA platform.  MCU
metrics use the same normalized SOL keys consumed by the existing roofline and
bottleneck components.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from triton_kernel_agent.kernel_backend import KernelBundle
from triton_kernel_agent.kernel_backend import extract_kernel_bundle
from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
)
from triton_kernel_agent.platform.nvidia import NvidiaWorkerRunner

_VALUE_WITH_OPTIONAL_UNIT_RE = re.compile(
    r"^(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?P<unit>%|GB/s|MB/s|ns|us|ms|s|cycle|GHz)?$"
)
_NORMALIZED_KEYS = {
    "compute throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "compute mp throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory throughput": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
}

_BENCHMARK_SCRIPT = r'''import importlib.util, json, sys, torch
import torch_musa

def load(path, name):
    parent = str(__import__("pathlib").Path(path).resolve().parent)
    if parent not in sys.path: sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

problem = load(sys.argv[1], "musa_problem")
mode = sys.argv[2]
warmup, repeat = int(sys.argv[3]), int(sys.argv[4])
init_inputs = problem.get_init_inputs() if hasattr(problem, "get_init_inputs") else []
if not isinstance(init_inputs, (list, tuple)): init_inputs = [init_inputs]
model = problem.Model(*init_inputs) if init_inputs else problem.Model()
model = model.to("musa")
inputs = problem.get_inputs()
if not isinstance(inputs, (list, tuple)): inputs = [inputs]
inputs = [x.to("musa") if isinstance(x, torch.Tensor) else x for x in inputs]
if mode == "kernel":
    fn = load(sys.argv[5], "musa_kernel").kernel_function
else:
    fn = model
for _ in range(warmup): fn(*inputs)
torch.musa.synchronize()
start = torch.musa.Event(enable_timing=True)
end = torch.musa.Event(enable_timing=True)
start.record()
for _ in range(repeat): fn(*inputs)
end.record()
torch.musa.synchronize()
print(json.dumps({"time_ms": start.elapsed_time(end) / repeat}))
'''


@dataclass
class MusaProfilerMetadata:
    kernel_file: str
    problem_file: str
    round_num: int
    timestamp: str
    mcu_version: str | None


@dataclass
class MusaProfilerResults:
    metrics_df: pd.DataFrame
    metrics: dict[str, Any]
    metadata: MusaProfilerMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "metadata": vars(self.metadata),
        }


def parse_mcu_metrics(stdout: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse MCU's human-readable table into raw and normalized metrics."""

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        columns = re.split(r"\s{2,}", line.strip())
        if len(columns) < 2 or not re.match(r"^[A-Za-z]", columns[0]):
            continue
        raw_name = columns[0].strip()
        # MCU 1.3 prints ``Metric Name | Metric Unit | Metric Value`` while
        # older/synthetic output may use ``Metric Name | Metric Value Unit``.
        if len(columns) >= 3 and re.fullmatch(
            r"-?[0-9][0-9,]*(?:\.[0-9]+)?", columns[-1]
        ):
            value_text = columns[-1]
            unit = columns[-2]
        else:
            match = _VALUE_WITH_OPTIONAL_UNIT_RE.fullmatch(columns[-1])
            if not match:
                continue
            value_text = match.group("value")
            unit = match.group("unit") or ""
        value = float(value_text.replace(",", ""))
        rows.append(
            {"Metric Name": raw_name, "Metric Value": value, "Metric Unit": unit}
        )
        slug = re.sub(r"[^a-z0-9]+", " ", raw_name.lower()).strip()
        raw_key = f"mcu::{slug.replace(' ', '_')}"
        if raw_key in metrics:
            # MCU can emit the same display name in different sections with
            # different units.  For example, SpeedOfLight reports Memory
            # Throughput as a percentage while MemoryWorkloadAnalysis reports
            # it in Gbyte/s.  Preserve both instead of silently overwriting the
            # first value.
            unit_slug = re.sub(r"[^a-z0-9]+", "_", unit.lower()).strip("_")
            raw_key = f"{raw_key}_{unit_slug or 'duplicate'}"
            suffix = 2
            candidate = raw_key
            while candidate in metrics:
                candidate = f"{raw_key}_{suffix}"
                suffix += 1
            raw_key = candidate
        metrics[raw_key] = value
        # Roofline SOL fields are percentages.  Do not map an absolute
        # throughput such as 696.03 Gbyte/s onto a pct_of_peak key.
        if unit == "%":
            for marker, normalized in _NORMALIZED_KEYS.items():
                if marker in slug:
                    metrics.setdefault(normalized, value)
    return pd.DataFrame(rows), metrics


class MusaVerifier(KernelVerifier):
    def __init__(self, log_dir: Path, logger: logging.Logger) -> None:
        self.log_dir = Path(log_dir)
        self.logger = logger

    def verify(
        self,
        kernel_code: str | dict[str, str] | KernelBundle,
        problem_file: Path,
        test_code: list[str],
    ) -> bool:
        from triton_kernel_agent.worker import VerificationWorker

        if isinstance(kernel_code, KernelBundle):
            payload = kernel_code
        elif isinstance(kernel_code, dict):
            payload = KernelBundle(kernel_code)
        else:
            # OptimizationManager represents native bundles using the same
            # FILE/fenced format sent to the model. Preserve all four files
            # instead of treating the rendered payload as kernel.py source.
            payload = extract_kernel_bundle(kernel_code) or KernelBundle(
                {"kernel.py": kernel_code}
            )
        verify_dir = self.log_dir / "initial_verify"
        verify_dir.mkdir(parents=True, exist_ok=True)
        # Generated and example tests commonly import ``problem``. Verification
        # runs in an isolated directory, so make that contract available there.
        shutil.copy2(problem_file, verify_dir / "problem.py")
        worker = VerificationWorker(
            worker_id=-1,
            workdir=verify_dir,
            log_dir=verify_dir,
            target_platform="musa",
            kernel_backend="musa",
            test_timeout_s=180,
        )
        success, _, error = worker.verify_with_refinement(
            payload, test_code, problem_file.read_text(encoding="utf-8"), 0
        )
        if not success:
            self.logger.error(
                "Initial MUSA kernel verification failed: %s", error[:500]
            )
        return success


class MusaBenchmarker(KernelBenchmarker):
    """Benchmark through the bundle's own test/benchmark entry point."""

    def __init__(
        self,
        log_dir: Path,
        logger: logging.Logger,
        benchmark_lock: Any,
        warmup: int = 5,
        repeat: int = 20,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.logger = logger
        self.benchmark_lock = benchmark_lock
        self.warmup = warmup
        self.repeat = repeat

    def _run(
        self, problem_file: Path, mode: str, kernel_file: Path | None = None
    ) -> float:
        artifacts = self.log_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        script = artifacts / "musa_benchmark.py"
        script.write_text(_BENCHMARK_SCRIPT, encoding="utf-8")
        cmd = [
            sys.executable,
            str(script),
            str(problem_file),
            mode,
            str(self.warmup),
            str(self.repeat),
        ]
        if kernel_file is not None:
            cmd.append(str(kernel_file))
        try:
            self.benchmark_lock.acquire()
            if kernel_file is not None:
                build = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=kernel_file.parent,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if build.returncode != 0:
                    self.logger.error(
                        "MUSA benchmark extension build failed: %s",
                        (build.stderr or build.stdout)[-4000:],
                    )
                    return float("inf")
            result = subprocess.run(
                cmd, cwd=artifacts, capture_output=True, text=True, timeout=600
            )
        except Exception as exc:
            self.logger.error("MUSA benchmark failed: %s", exc)
            return float("inf")
        finally:
            try:
                self.benchmark_lock.release()
            except Exception:
                pass
        if result.returncode != 0:
            self.logger.error("MUSA benchmark failed: %s", result.stderr[-2000:])
            return float("inf")
        try:
            return float(json.loads(result.stdout.strip().splitlines()[-1])["time_ms"])
        except (IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.logger.error("Invalid MUSA benchmark output: %s", exc)
            return float("inf")

    def benchmark_kernel(self, kernel_code: Any, problem_file: Path) -> Any:
        # OptimizationManager passes a rendered bundle and expects a float;
        # OptimizationWorker passes the materialized kernel.py and expects the
        # standard worker result mapping.
        worker_call = isinstance(kernel_code, Path)
        if worker_call:
            kernel_file = kernel_code
            required = ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
            if not all((kernel_file.parent / name).is_file() for name in required):
                self.logger.error("MUSA benchmark requires a four-file bundle")
                return {"time_ms": float("inf"), "speedup": 0.0, "ptx_hash": None}
            time_ms = self._run(problem_file, "kernel", kernel_file)
            return {"time_ms": time_ms, "speedup": 1.0, "ptx_hash": None}

        bundle = (
            kernel_code
            if isinstance(kernel_code, KernelBundle)
            else KernelBundle(kernel_code)
            if isinstance(kernel_code, dict)
            else extract_kernel_bundle(kernel_code)
        )
        if bundle is None:
            self.logger.error("MUSA benchmark requires a multi-file kernel bundle")
            return float("inf")
        artifacts = self.log_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        for name, content in bundle.files.items():
            (artifacts / name).write_text(content, encoding="utf-8")
        return self._run(problem_file, "kernel", artifacts / "kernel.py")

    def benchmark_pytorch(self, problem_file: Path) -> dict[str, Any]:
        """Worker-facing eager baseline API."""

        return {"time_ms": self.benchmark_reference(problem_file)}

    def benchmark_reference(self, problem_file: Path) -> float:
        return self._run(problem_file, "reference")

    def benchmark_reference_compiled(self, problem_file: Path) -> float:
        return float("inf")


class MusaWorkerRunner(NvidiaWorkerRunner):
    """Use the upstream bounded multi-process scheduler with MUSA workers.

    The scheduler itself is accelerator-neutral. ``target_platform=musa`` and
    MUSA registry component names are carried in ``worker_kwargs``.
    """

    def __init__(
        self, *args: Any, worker_kwargs: dict[str, Any], **kwargs: Any
    ) -> None:
        worker_kwargs = {
            **worker_kwargs,
            "target_platform": "musa",
            "kernel_backend": "musa",
        }
        super().__init__(*args, worker_kwargs=worker_kwargs, **kwargs)

    def run_workers(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        # torch_musa honors this variable; the inherited scheduler retains the
        # upstream bounded joins, queue draining, and per-device lock behavior.
        os.environ.setdefault(
            "MUSA_VISIBLE_DEVICES", ",".join(str(device) for device in self.gpu_ids)
        )
        return super().run_workers(*args, **kwargs)


class MusaAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        try:
            import torch
            import torch_musa  # noqa: F401

            props = torch.musa.get_device_properties(torch.musa.current_device())
            return {
                "name": device_name or torch.musa.get_device_name(),
                "multi_processor_count": getattr(props, "multi_processor_count", None),
                "total_memory": getattr(props, "total_memory", None),
                "platform": "musa",
            }
        except Exception as exc:
            raise RuntimeError(
                f"Unable to query MUSA device properties: {exc}"
            ) from exc


class MusaBottleneckAnalyzer(BottleneckAnalyzerBase):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        openai_model: str = "gpt-5",
        gpu_name: str | None = None,
        num_bottlenecks: int = 1,
        roofline_config: Any | None = None,
        **_: Any,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.log_dir = log_dir
        self.openai_model = openai_model
        self.gpu_name = gpu_name
        self.num_bottlenecks = num_bottlenecks
        self._delegate: Any | None = None
        # OptimizationOrchestrator accesses this attribute directly before it
        # asks the LLM bottleneck analyzer for recommendations.
        from triton_kernel_agent.platform.nvidia import NvidiaRooflineAnalyzer

        self.roofline = NvidiaRooflineAnalyzer(
            logger=self.logger, roofline_config=roofline_config
        )

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (  # noqa: E501
                BottleneckAnalyzer,
            )
            from utils.providers import get_model_provider

            specs = MusaAcceleratorSpecsProvider().get_specs(self.gpu_name)
            self._delegate = BottleneckAnalyzer(
                provider=get_model_provider(self.openai_model),
                model=self.openai_model,
                gpu_specs=specs,
                logs_dir=self.log_dir,
                logger=self.logger,
                num_bottlenecks=self.num_bottlenecks,
            )
        return self._delegate

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        return self._get_delegate().analyze(
            kernel_code, ncu_metrics, round_num, roofline_result
        )


class MusaKernelProfiler(KernelProfilerBase):
    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        artifacts_dir: Path | None = None,
        mcu_bin_path: str | None = None,
        ncu_bin_path: str | None = None,
        mcu_timeout_seconds: int = 300,
        mcu_sampling_interval: int | None = None,
        profiling_semaphore: Any | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.log_dir = Path(log_dir or ".").resolve()
        self.artifacts_dir = Path(artifacts_dir or self.log_dir / "artifacts").resolve()
        # ``ncu_bin_path`` is the existing cross-platform OptimizationWorker
        # option. Accept it as an alias so a non-standard MCU installation can
        # be configured without adding a MUSA-only constructor argument to the
        # whole manager stack.
        self.mcu_bin_path = (
            mcu_bin_path or ncu_bin_path or os.getenv("MCU_BIN") or "mcu"
        )
        self.timeout = mcu_timeout_seconds
        configured_interval = (
            mcu_sampling_interval
            if mcu_sampling_interval is not None
            else os.getenv("MUSA_MCU_SAMPLING_INTERVAL")
        )
        self.sampling_interval = (
            int(configured_interval) if configured_interval not in (None, "") else None
        )
        if self.sampling_interval is not None and not 1 <= self.sampling_interval <= 65535:
            raise ValueError("MCU sampling interval must be in [1, 65535]")
        self.semaphore = profiling_semaphore

    def _profile_command(
        self,
        wrapper: Path,
        report: Path,
        sampling_interval: int | None = None,
    ) -> list[str]:
        """Build the MCU command and keep each binary report for diagnosis."""

        interval = (
            sampling_interval
            if sampling_interval is not None
            else self.sampling_interval
        )
        command = [
            self.mcu_bin_path,
            "--sections",
            "SpeedOfLight",
            "--page",
            "details",
            "--force-overwrite",
            "--output",
            str(report),
            sys.executable,
            str(wrapper),
        ]
        if interval is not None:
            output_index = command.index("--output")
            command[output_index:output_index] = [
                "--sampling-interval",
                str(interval),
            ]
        return command

    @staticmethod
    def _has_terminal_pfm_failure(output: str) -> bool:
        """Return whether MCU failed inside PFM collection.

        MCU 1.3 may still exit with status 0 after reporting a PFM buffer or
        dump overlap and then print ``No kernels were profiled``.  Replaying
        the same large workload with nearby sampling intervals is expensive
        and, more importantly, cannot repair that collector failure.
        """

        lowered = output.lower()
        return any(
            marker in lowered
            for marker in (
                "pfm buffer overflow",
                "pfm dump overlap",
                "pfm ap overlap",
            )
        )

    @staticmethod
    def _has_terminal_command_failure(output: str) -> bool:
        """Return whether MCU rejected the profiler command itself.

        MCU 1.1.1 may print a command-line error while still exiting with status
        zero.  Treat those messages as terminal so the caller does not mistake
        them for an empty profile and retry the same invalid command with other
        sampling intervals.
        """

        lowered = output.lower()
        return any(
            marker in lowered
            for marker in (
                "unrecognised option",
                "unrecognized option",
                "unknown option",
                "invalid option",
                "missing argument for option",
            )
        )

    def _save_raw_attempt(
        self,
        round_num: int,
        attempt: int,
        cmd: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> Path:
        """Persist MCU console output so intermittent collection can be diagnosed."""

        prefix = self.log_dir / f"round{round_num:03d}_mcu_attempt{attempt}"
        prefix.with_suffix(".command.txt").write_text(
            shlex.join(cmd) + "\n", encoding="utf-8"
        )
        prefix.with_suffix(".stdout.txt").write_text(
            result.stdout or "", encoding="utf-8"
        )
        prefix.with_suffix(".stderr.txt").write_text(
            result.stderr or "", encoding="utf-8"
        )
        return prefix

    @staticmethod
    def _profile_hash(kernel_file: Path, problem_file: Path) -> str:
        digest = hashlib.sha256()
        for name in ("kernel.py", "binding.cpp", "kernel.mu", "setup.py"):
            path = kernel_file.parent / name
            if path.is_file():
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
        digest.update(b"problem.py\0")
        digest.update(problem_file.read_bytes())
        return digest.hexdigest()

    def _cache_path(self, kernel_file: Path, problem_file: Path) -> Path:
        root = Path(
            os.getenv(
                "MUSA_MCU_CACHE_DIR",
                str(Path.cwd() / ".kernelagent" / "mcu_profiles"),
            )
        )
        return root / f"{self._profile_hash(kernel_file, problem_file)}.json"

    def _load_cached_profile(
        self, kernel_file: Path, problem_file: Path, round_num: int
    ) -> MusaProfilerResults | None:
        path = self._cache_path(kernel_file, problem_file)
        if not path.is_file():
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            metrics = cached["metrics"]
            if not isinstance(metrics, dict) or not metrics:
                return None
            self.logger.warning(
                "Using source-hash-matched MCU profile cache after live profiling failed: %s",
                path,
            )
            return MusaProfilerResults(
                pd.DataFrame(cached.get("rows", [])),
                metrics,
                MusaProfilerMetadata(
                    str(kernel_file),
                    str(problem_file),
                    round_num,
                    datetime.now(timezone.utc).isoformat(),
                    cached.get("mcu_version"),
                ),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_cached_profile(
        self, kernel_file: Path, problem_file: Path, parsed: MusaProfilerResults
    ) -> None:
        path = self._cache_path(kernel_file, problem_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "source_hash": self._profile_hash(kernel_file, problem_file),
                    "metrics": parsed.metrics,
                    "rows": parsed.metrics_df.to_dict(orient="records"),
                    "mcu_version": parsed.metadata.mcu_version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _version(self) -> str | None:
        try:
            result = subprocess.run(
                [self.mcu_bin_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return (
                result.stdout.strip().splitlines()[0]
                if result.returncode == 0
                else None
            )
        except Exception:
            return None

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 3,
    ) -> MusaProfilerResults | None:
        from triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory import (  # noqa: E501
            NCUWrapperFactory,
        )

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        setup_file = kernel_file.parent / "setup.py"
        if setup_file.is_file():
            try:
                build = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=kernel_file.parent,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if build.returncode != 0:
                    self.logger.warning(
                        "MCU extension build failed: %s",
                        (build.stderr or build.stdout)[-4000:],
                    )
                    return None
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.logger.warning("MCU extension build failed: %s", exc)
                return None
        wrapper = NCUWrapperFactory(self.logger).create_ncu_wrapper(
            kernel_file,
            problem_file,
            self.artifacts_dir,
            target_platform="musa",
        )
        acquired = self.semaphore is None or self.semaphore.acquire(timeout=900)
        if not acquired:
            return None
        try:
            # Use MCU's documented default first.  For non-PFM failures, retry
            # with progressively coarser sampling.  Explicit configuration is
            # authoritative, but retries must only increase the interval.
            retry_intervals: list[int | None] = [self.sampling_interval]
            if self.sampling_interval is None:
                retry_intervals.extend([1000, 10000])
            else:
                retry_intervals.extend(
                    [
                        max(1, min(65535, self.sampling_interval * 2)),
                        max(1, min(65535, self.sampling_interval * 10)),
                    ]
                )
            for attempt, sampling_interval in enumerate(
                retry_intervals[: max(1, max_retries)]
            ):
                try:
                    report = self.log_dir / (
                        f"round{round_num:03d}_mcu_attempt{attempt + 1}.mcu-rep"
                    )
                    cmd = self._profile_command(
                        wrapper, report, sampling_interval=sampling_interval
                    )
                    result = subprocess.run(
                        cmd,
                        cwd=self.artifacts_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                    raw_prefix = self._save_raw_attempt(
                        round_num, attempt + 1, cmd, result
                    )
                    combined_output = "\n".join(
                        part for part in (result.stdout, result.stderr) if part
                    )
                    if result.returncode != 0:
                        raise RuntimeError(combined_output)
                    if "application returned an error code" in combined_output.lower():
                        self.logger.error(
                            "MCU target application failed before profiling; "
                            "sampling retries cannot fix it. Raw output: %s.*: %s",
                            raw_prefix,
                            combined_output[-2000:],
                        )
                        return None
                    if self._has_terminal_pfm_failure(combined_output):
                        self.logger.error(
                            "MCU PFM collection failed (buffer/dump overlap); "
                            "stopping sampling retries and falling back. "
                            "Raw output: %s.*",
                            raw_prefix,
                        )
                        return self._load_cached_profile(
                            kernel_file, problem_file, round_num
                        )
                    if self._has_terminal_command_failure(combined_output):
                        self.logger.error(
                            "MCU rejected the profiler command; sampling retries "
                            "cannot fix it. Raw output: %s.*: %s",
                            raw_prefix,
                            combined_output[-2000:],
                        )
                        return None
                    frame, metrics = parse_mcu_metrics(combined_output)
                    if not metrics:
                        raise ValueError(
                            "MCU output contained no parseable metrics; "
                            f"raw output saved to {raw_prefix}.*"
                        )
                    # The optimization pipeline consumes profiler metrics as
                    # ``kernel_name -> metric mapping`` (the same shape NCU
                    # returns), while parse_mcu_metrics intentionally exposes
                    # a flat mapping for standalone callers.
                    nested_metrics = {kernel_file.stem: metrics}
                    parsed = MusaProfilerResults(
                        frame,
                        nested_metrics,
                        MusaProfilerMetadata(
                            str(kernel_file),
                            str(problem_file),
                            round_num,
                            datetime.now(timezone.utc).isoformat(),
                            self._version(),
                        ),
                    )
                    output = self.log_dir / f"round{round_num:03d}_mcu_metrics.json"
                    output.write_text(
                        json.dumps(parsed.to_dict(), indent=2), encoding="utf-8"
                    )
                    self._save_cached_profile(kernel_file, problem_file, parsed)
                    return parsed
                except (subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                    self.logger.warning(
                        "MCU profiling attempt %d (sampling interval %s) failed: %s",
                        attempt + 1,
                        sampling_interval if sampling_interval is not None else "default",
                        exc,
                    )
            return self._load_cached_profile(kernel_file, problem_file, round_num)
        finally:
            if self.semaphore is not None:
                self.semaphore.release()
