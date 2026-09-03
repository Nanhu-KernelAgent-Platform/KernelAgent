#!/usr/bin/env python3
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

"""Example demonstrating multi-strategy kernel optimization using OptimizationManager.

This script shows how to use the OptimizationManager to optimize a Triton kernel
using different search strategies:
- beam_search: Maintain top-N kernels, explore M bottlenecks each
- greedy: Simple single-best optimization with early termination
- noop: Dry-run without GPU hardware (returns initial kernel unchanged)

The OptimizationManager orchestrates parallel workers and persists optimization
history to a JSON database for analysis and resumption.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from triton_kernel_agent.opt_manager import OptimizationManager
from triton_kernel_agent.kernel_backend import KernelBundle, extract_kernel_bundle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# Hardcoded config directory relative to this script.
_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _discover_strategies() -> list[str]:
    """Discover available strategies from yaml files in the configs directory."""
    return sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))


def _run_strategy(
    strategy: str,
    kernel_code: str,
    problem_file: Path,
    test_code: str,
    log_dir: Path,
    max_rounds: int | None = None,
) -> dict:
    """Run a single strategy using its config file."""
    config_path = _CONFIGS_DIR / f"{strategy}.yaml"
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"{strategy.upper()} OPTIMIZATION")
    print("=" * 80)
    print(f"Config: {config_path}")

    manager = OptimizationManager(
        config=str(config_path),
        log_dir=log_dir / strategy,
        database_path=log_dir / strategy / "program_db.json",
    )

    return manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
        max_rounds=max_rounds,
    )


def summarize_result(result: dict) -> dict:
    """Return a compact, serializable optimization outcome."""

    initial = float(result.get("initial_kernel_time_ms", float("inf")))
    best = float(result.get("best_time_ms", float("inf")))
    candidate_count = len(result.get("top_kernels") or [])
    improved = (
        candidate_count > 0
        and math.isfinite(initial)
        and math.isfinite(best)
        and best < initial
    )
    improvement_pct = (
        (initial - best) / initial * 100.0
        if math.isfinite(initial) and initial > 0 and math.isfinite(best)
        else None
    )
    if not result.get("success"):
        status = "FAILED"
    elif candidate_count == 0:
        status = "DEGRADED"
    elif improved:
        status = "SUCCESS"
    else:
        status = "NO_GAIN"
    return {
        "status": status,
        "success": bool(result.get("success")),
        "candidate_count": candidate_count,
        "initial_time_ms": initial if math.isfinite(initial) else None,
        "best_time_ms": best if math.isfinite(best) else None,
        "pytorch_baseline_ms": result.get("pytorch_baseline_ms"),
        "total_rounds": int(result.get("total_rounds", 0)),
        "improvement_pct": improvement_pct,
    }


def write_best_kernel(result: dict, strategy_name: str, kernel_dir: Path) -> dict:
    """Persist the serialized best kernel and a materialized native bundle."""

    paths: dict[str, str | None] = {
        "serialized_kernel": None,
        "best_bundle_dir": None,
    }
    kernel_code = result.get("kernel_code")
    if not kernel_code:
        return paths
    output_file = kernel_dir / f"optimized_kernel_{strategy_name.lower()}.py"
    output_file.write_text(kernel_code, encoding="utf-8")
    paths["serialized_kernel"] = str(output_file)
    bundle = extract_kernel_bundle(kernel_code)
    if bundle is not None:
        bundle_dir = kernel_dir / "best_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for name, content in bundle.files.items():
            (bundle_dir / name).write_text(content, encoding="utf-8")
        paths["best_bundle_dir"] = str(bundle_dir)
    return paths


def print_result(result: dict, strategy_name: str, kernel_dir: Path) -> dict:
    """Print optimization result and save the best kernel."""
    if result["success"]:
        print()
        print("=" * 80)
        print(f"{strategy_name} OPTIMIZATION SUCCESSFUL!")
        print("=" * 80)
        print(f"Best time: {result['best_time_ms']:.4f} ms")
        print(f"Total rounds: {result['total_rounds']}")
        print(f"Top kernels found: {len(result['top_kernels'])}")

        # Show top kernels summary
        if result["top_kernels"]:
            print("\nTop kernels performance:")
            for i, kernel in enumerate(result["top_kernels"][:3], 1):
                print(f"  {i}. {kernel['time_ms']:.4f}ms (gen {kernel['generation']})")

        paths = write_best_kernel(result, strategy_name, kernel_dir)
        if paths["serialized_kernel"]:
            print(f"\nSaved optimized kernel to: {paths['serialized_kernel']}")
        if paths["best_bundle_dir"]:
            print(f"Materialized best bundle at: {paths['best_bundle_dir']}")
    else:
        print()
        print("=" * 80)
        print(f"{strategy_name} OPTIMIZATION FAILED")
        print("=" * 80)
        print("Check logs for details")
        paths = {"serialized_kernel": None, "best_bundle_dir": None}
    return {**summarize_result(result), **paths}


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Optimize Triton kernels using different search strategies"
    )
    parser.add_argument(
        "--strategy",
        default="beam_search",
        help="Optimization strategy to use, or 'all' to run every config in configs/ (default: beam_search)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum optimization rounds (default: 5)",
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Directory containing kernel.py, problem.py, and test.py",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Write a compact machine-readable result (default: <kernel-dir>/optimization_result.json)",
    )

    args = parser.parse_args()

    # Setup paths
    kernel_dir = args.kernel_dir.resolve()
    kernel_file = kernel_dir / "input.py"  # Use input.py as initial
    problem_file = kernel_dir / "problem.py"
    test_file = kernel_dir / "test.py"
    log_dir = kernel_dir / "opt_manager_logs"

    if not problem_file.exists():
        print(f"ERROR: problem.py not found in {kernel_dir}")
        sys.exit(1)

    if not test_file.exists():
        print(f"ERROR: test.py not found in {kernel_dir}")
        sys.exit(1)

    # Read source files
    bundle_names = ("kernel.py", "binding.cpp", "kernel.mu", "setup.py")
    if all((kernel_dir / name).exists() for name in bundle_names):
        kernel_code = KernelBundle(
            {name: (kernel_dir / name).read_text() for name in bundle_names}
        ).render_for_prompt()
    else:
        kernel_code = kernel_file.read_text()
    test_code = test_file.read_text()

    # Print header
    print("=" * 80)
    print("OptimizationManager - Multi-Strategy Kernel Optimization")
    print("=" * 80)
    print(f"Kernel file: {kernel_file}")
    print(f"Problem file: {problem_file}")
    print(f"Strategy: {args.strategy}")
    print(f"Max rounds: {args.max_rounds}")
    print(f"Log directory: {log_dir}")

    # Run selected strategy
    if args.strategy == "all":
        results = {}
        for strategy in _discover_strategies():
            results[strategy] = _run_strategy(
                strategy,
                kernel_code,
                problem_file,
                test_code,
                log_dir,
                max_rounds=args.max_rounds,
            )
            print_result(results[strategy], strategy.upper(), kernel_dir)

        # Compare results
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        for name, result in results.items():
            status = "+" if result["success"] else "-"
            time_str = f"{result['best_time_ms']:.4f}ms" if result["success"] else "N/A"
            print(f"  {status} {name:15} - Best: {time_str}")
    else:
        result = _run_strategy(
            args.strategy,
            kernel_code,
            problem_file,
            test_code,
            log_dir,
            max_rounds=args.max_rounds,
        )
        summary = print_result(result, args.strategy.upper(), kernel_dir)
        result_json = (args.result_json or kernel_dir / "optimization_result.json").resolve()
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Optimization result JSON: {result_json}")


if __name__ == "__main__":
    main()
