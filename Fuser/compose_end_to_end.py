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
"""
Compose an end-to-end backend kernel for the original KernelBench problem by
leveraging:
  1) Fuser's subgraphs JSON (decomposition + shapes)
  2) KernelAgent-generated backend kernels for those subgraphs

We call an LLM to synthesize a final composed kernel that matches the original
problem semantics, returning one complete Python file that exposes
`kernel_function(...)`. Triton uses a single Python file; native MUSA uses a
validated four-file extension bundle and a self-test.

Usage:
  python -m Fuser.compose_end_to_end \
      --problem /abs/path/to/kernelbench_problem.py \
      --subgraphs /abs/path/to/subgraphs.json \
      --kernels-summary /abs/path/to/kernels_out/summary.json \
      [--model gpt-5] [--out-dir ./compose_out] [--verify]

Notes:
- Requires an available LLM provider configured via KernelAgent providers
  (e.g., OPENAI_API_KEY for OpenAI models).
- Writes composed Python file to <out-dir>/composed_kernel.py and a
composition summary JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from triton_kernel_agent.platform_config import (
    get_platform,
    get_platform_choices,
    PlatformConfig,
)
from triton_kernel_agent.kernel_backend import (
    KernelBundle,
    get_kernel_backend,
    get_kernel_backend_choices,
    KernelBackendConfig,
    extract_kernel_bundle,
)

# Reuse KernelAgent provider stack for LLM calls
try:
    from utils.providers.models import get_model_provider
except Exception:
    get_model_provider = None  # type: ignore

# Reuse extractor to capture clean python code blocks
from .code_extractor import extract_single_python_file

# Reuse Fuser runner for optional verification
from .runner import run_candidate, run_bundle_candidate


@dataclass
class KernelItem:
    subgraph_id: str
    kernel_path: Path
    code: str
    files: dict[str, str]
    kernel_backend: str = "triton"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_kernels_from_summary(
    summary_path: Path, expected_backend: str = "triton"
) -> list[KernelItem]:
    data = json.loads(_read_text(summary_path))
    if not isinstance(data, list):
        raise SystemExit("kernels summary must be a JSON array (from dispatch step)")
    items: list[KernelItem] = []
    summary_dir = summary_path.parent
    for it in data:
        if not isinstance(it, dict):
            continue
        if not it.get("success"):
            # Skip failed generations
            continue
        sid = str(it.get("id", ""))
        item_backend = str(it.get("kernel_backend") or "triton")
        if item_backend != expected_backend:
            raise SystemExit(
                f"summary backend mismatch for {sid or '<unknown>'}: "
                f"expected {expected_backend}, got {item_backend}"
            )
        kpath_str = it.get("kernel_path") or ""
        if not sid or not kpath_str:
            continue
        kpath = Path(kpath_str)
        # If path is relative, resolve it relative to summary.json location
        if not kpath.is_absolute():
            kpath = summary_dir / kpath
        if not kpath.is_file():
            continue
        artifact_dir = it.get("artifact_dir")
        files: dict[str, str] = {}
        file_map = it.get("files")
        if isinstance(file_map, dict):
            for name, raw_path in file_map.items():
                path = Path(str(raw_path))
                if not path.is_absolute():
                    path = summary_dir / path
                if path.is_file():
                    files[str(name)] = _read_text(path)
        if not files and artifact_dir:
            adir = Path(str(artifact_dir))
            if not adir.is_absolute():
                adir = summary_dir / adir
            if adir.is_dir():
                for path in adir.iterdir():
                    if path.is_file() and path.name not in {"problem.txt"}:
                        files[path.name] = _read_text(path)
        if not files:
            files = {"kernel.py": _read_text(kpath)}
        code = files.get("kernel.py", _read_text(kpath))
        items.append(
            KernelItem(
                subgraph_id=sid,
                kernel_path=kpath,
                code=code,
                files=files,
                kernel_backend=item_backend,
            )
        )
    if not items:
        raise SystemExit("no successful kernels in summary.json")
    return items


def _summarize_subgraphs_for_prompt(subgraphs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for it in subgraphs:
        sid = str(it.get("id", "unknown"))
        typ = str(it.get("type", ""))
        layout = it.get("data_layout") or "NCHW"
        dtype = it.get("dtype") or "float32"
        inputs = it.get("inputs")
        in_shape = it.get("input_shape")
        out_shape = it.get("output_shape")
        ops = it.get("ops") or []
        # Compact rendering
        shapes_line = (
            f"inputs={inputs if inputs is not None else in_shape}, output={out_shape}"
        )
        lines.append(
            f"- ID={sid} type={typ} layout={layout} dtype={dtype} {shapes_line}"
        )
        try:
            ops_short = json.dumps(ops)[:400]
        except Exception:
            ops_short = str(ops)[:400]
        lines.append(f"  ops={ops_short}")
    return "\n".join(lines)


def _build_composition_prompt(
    problem_code: str,
    subgraphs: list[dict[str, Any]],
    kernel_items: list[KernelItem],
    target_platform: PlatformConfig,
) -> str:
    """Create a single user message to instruct composition by the LLM."""
    # Provide a succinct summary of subgraphs up front
    sg_summary = _summarize_subgraphs_for_prompt(subgraphs)

    # Include only essential snippets from each kernel to keep token usage sane
    # We include full files for now; callers can trim by model limits.
    kernels_section_parts: list[str] = []
    for ki in kernel_items:
        kernels_section_parts.append(
            f"### Subgraph {ki.subgraph_id}\n```python\n" + ki.code + "\n```\n"
        )
    kernels_section = "\n".join(kernels_section_parts)
    # Platform-specific guidance
    platform_guidance = target_platform.guidance_block

    guidance = textwrap.dedent(
        f"""
        You are given:
        - The original problem file (PyTorch module and helpers).
        - A decomposition of the model into fusable subgraphs with exact shapes.
        - Working Triton kernels generated for some subgraphs.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}
        {platform_guidance}

        Task:
        - Compose an end-to-end Triton implementation that matches the original
          model's forward pass for the provided shapes. You may inline, adapt,
          or reuse the given subgraph kernels. Prefer fusing into as few kernel
          launches as possible while preserving exact numerical semantics.

        Hard requirements:
        - Return ONE complete Python file only, fenced as a single ```python block.
        - Allocate inputs, weights, intermediates, and outputs on device='{target_platform.device_string}' and keep them there throughout forward/verification.
        - CPU is acceptable only for metadata, scalars, and export serialization—avoid `.cpu()` or `.to('cpu')` on compute tensors.
        - Provide at least one @triton.jit kernel and a top-level Python wrapper
          named kernel_function(...). This wrapper must accept the same primary
          input tensor(s) as the model and any required weights/biases with shapes
          implied by the problem; it should orchestrate Triton kernel(s) and
          return the final output tensor.
        - No PyTorch math path: kernel_function MUST compute the final outputs
          using your Triton kernels only. Do NOT implement or fall back to
          torch.nn / torch.nn.functional / torch.* ops
          sigmoid, etc.) for producing the final result. Using PyTorch for
          reference comparisons is allowed only inside the self-test.
        - Use the data layout and dtype semantics indicated by subgraphs, defaulting
          to NCHW + float32 if unspecified. Respect stride/padding/dilation/groups,
          and exact op order.
        - Numerical equivalence: include a self-test (test_kernel or run_tests)
          that compares your Triton-based result to a PyTorch reference computed
          from the original problem code below (use get_init_inputs() and
          get_inputs() if present to instantiate the Model). The test must print
          'PASS' on success and exit with code 0. Use allclose with rtol<=1e-3,
          atol<=1e-3 for fp32; for fp16/bf16 allow up to 2e-2.
        - No imports beyond torch, triton, triton.language as tl, and stdlib. No I/O.
        - Do NOT monkey-patch PyTorch device functions or torch.cuda.is_available()
        - Do NOT manipulate TRITON_BACKENDS environment variable
        - Do NOT disable or mock XPU/CUDA drivers

        Implementation tips:
        - If merging multiple subgraphs, ensure intermediate tensor shapes match.
        - Hoist constant weights or parameters to avoid reloading per block.
        - Use tl.load/tl.store with masks for boundary conditions.
        - Favor coalesced memory access; tile by blocks; compute grid from shape.
        - Common Triton pitfalls to avoid:
          * Do NOT call tl.broadcast on Python scalars; tl.maximum(x, 0.0) works.
          * Prefer scalar constants directly in elementwise ops (no explicit broadcast needed).
          * Keep BLOCK_SIZE power-of-two; mask stores at tail.
        """
    ).strip()

    user_lines: list[str] = []
    user_lines.append(guidance)
    user_lines.append("")
    user_lines.append("SUBGRAPHS (summary):")
    user_lines.append(sg_summary)
    user_lines.append("")
    user_lines.append("ORIGINAL PROBLEM FILE:")
    user_lines.append("```python")
    user_lines.append(problem_code)
    user_lines.append("```")
    user_lines.append("")
    user_lines.append("SUBGRAPH KERNELS (reference implementations):")
    user_lines.append(kernels_section)
    user_lines.append("")
    user_lines.append(
        "Return only one fenced Python code block with your final composed implementation."
    )
    return "\n".join(user_lines)


def _build_refinement_prompt(
    problem_code: str,
    subgraphs: list[dict[str, Any]],
    kernel_items: list[KernelItem],
    previous_code: str,
    error_info: dict[str, str],
    target_platform: PlatformConfig,
) -> str:
    """Prompt the LLM to refine the previously produced code based on errors."""
    err_tail = error_info.get("stderr_tail", "")
    out_tail = error_info.get("stdout_tail", "")

    guidance = textwrap.dedent(
        f"""
        You previously produced a composed Triton implementation, but it failed
        to run/compile. Analyze the ERROR_CONTEXT below and re-emit the entire
        corrected single-file implementation as one ```python block.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}

        Requirements remain the same. Additionally:
        - Fix any Triton compilation/runtime errors. For scalar constants in
          elementwise ops (e.g., ReLU), do not use tl.broadcast. Use direct
          scalars like 0.0 in tl.maximum(x, 0.0).
        - Keep function name kernel_function(...) unchanged and retain the
          self-test that prints PASS on success and exits 0.
        - Do NOT reintroduce any PyTorch math path in kernel_function. The final
          outputs must be computed via your Triton kernels only (no fallback to
          torch.nn / torch.nn.functional ops).
        - Return the complete corrected file; do not send diffs.
        """
    ).strip()

    lines: list[str] = []
    lines.append(guidance)
    lines.append("")
    lines.append("ERROR_CONTEXT (stderr tail):\n```\n" + err_tail + "\n```")
    if out_tail.strip():
        lines.append("STDOUT tail:\n```\n" + out_tail + "\n```")
    lines.append("")
    lines.append("ORIGINAL PROBLEM FILE:\n```python\n" + problem_code + "\n```")
    lines.append("")
    lines.append("SUBGRAPHS (summary):\n" + _summarize_subgraphs_for_prompt(subgraphs))
    lines.append("")
    # Keep previous attempt for reference
    lines.append("PREVIOUS_ATTEMPT:\n```python\n" + previous_code + "\n```")
    lines.append("")
    lines.append(
        "Return only one fenced Python code block with the corrected implementation."
    )
    return "\n".join(lines)


def _render_bundle_for_prompt(item: KernelItem) -> str:
    return KernelBundle(item.files).render_for_prompt()


def _build_musa_composition_prompt(
    problem_code: str,
    subgraphs: list[dict[str, Any]],
    kernel_items: list[KernelItem],
    target_platform: PlatformConfig,
) -> str:
    sections = []
    for item in kernel_items:
        sections.append(
            f"### Subgraph {item.subgraph_id}\n{_render_bundle_for_prompt(item)}"
        )
    return textwrap.dedent(
        f"""
        You are given the original PyTorch problem, a list of exact fusable
        subgraphs, and native MUSA bundles that implement individual subgraphs.

        TARGET PLATFORM: {target_platform.name}
        DEVICE STRING: {target_platform.device_string}

        Compose one end-to-end native MUSA implementation for the original
        model. You may reuse the subgraph algorithms, but emit one coherent
        extension rather than importing multiple subgraph setup modules.

        Hard requirements:
        - Return exactly four FILE blocks and no prose: kernel.py, binding.cpp,
          kernel.mu, and setup.py.
        - kernel.py must expose kernel_function with the original input/weight
          contract and load the extension built by setup.py.
        - binding.cpp contains only PyTorch binding and launch dispatch glue.
        - All numerical work must be implemented in kernel.mu. Do not replace
          it with torch.nn, torch.nn.functional, or eager tensor math.
        - setup.py must use torch_musa.utils.musa_extension.MUSAExtension and
          BuildExtension.
        - Keep tensors and synchronization on torch.musa and never use
          torch.cuda APIs.
        - When kernel.py is executed directly it must run a self-test based on
          get_init_inputs()/get_inputs() from the original problem. Inline the
          required reference code in kernel.py; do not import an external
          problem file. Print PASS on success and exit with code 0.

        ORIGINAL PROBLEM:
        ```python
        {problem_code}
        ```

        SUBGRAPHS:
        {json.dumps(subgraphs, indent=2)}

        SUBGRAPH BUNDLES:
        {chr(10).join(sections)}

        Return only the four FILE blocks.
        """
    ).strip()


def _build_musa_refinement_prompt(
    problem_code: str,
    subgraphs: list[dict[str, Any]],
    previous_bundle: KernelBundle,
    error_info: dict[str, str],
    target_platform: PlatformConfig,
) -> str:
    return textwrap.dedent(
        f"""
        Repair the native MUSA bundle below after its self-test failed. Return
        the complete corrected bundle as exactly four FILE blocks and no prose.

        TARGET PLATFORM: {target_platform.name}
        ERROR STDOUT:
        ```
        {error_info.get("stdout", "")[-4000:]}
        ```
        ERROR STDERR:
        ```
        {error_info.get("stderr", "")[-4000:]}
        ```

        Preserve kernel_function's public contract. Keep computation in
        kernel.mu, retain MUSAExtension in setup.py, and keep all tensors on
        torch.musa. Inline the original-problem reference needed by the
        self-test; kernel.py must print PASS on success.

        ORIGINAL PROBLEM:
        ```python
        {problem_code}
        ```

        SUBGRAPHS:
        {json.dumps(subgraphs, indent=2)}

        CURRENT BUNDLE:
        {previous_bundle.render_for_prompt()}

        Return only FILE blocks for kernel.py, binding.cpp, kernel.mu, setup.py.
        """
    ).strip()


def _auto_patch_common_triton_issues(
    code: str, target_platform: PlatformConfig
) -> tuple[str, bool]:
    """Apply tiny safe textual patches for known Triton pitfalls.

    - Replace tl.broadcast(0.0, ...) or tl.broadcast(1.0, ...) with scalar constants.
    Returns (patched_code, changed).
    """
    patched = code
    changed = False
    # Simple heuristics; keep conservative
    patterns = [
        ("tl.broadcast(0.0", "0.0"),
        ("tl.broadcast(1.0", "1.0"),
        ("tl.broadcast(0,", "0.0"),
        ("tl.broadcast(1,", "1.0"),
    ]
    for old, new in patterns:
        if old in patched:
            patched = patched.replace(old, new)
            changed = True
    # Remove cuda paterns
    cuda_hacks = target_platform.cuda_hacks_to_strip
    if cuda_hacks:
        lines = patched.split("\n")
        filtered_lines = []
        skip_until_blank = False
        for line in lines:
            # Check if we're in a block to skip
            if skip_until_blank:
                if line.strip() == "":
                    skip_until_blank = False
                continue
            # Check if line contains any CUDA hack pattern
            if any(hack in line for hack in cuda_hacks):
                changed = True
                # Start skipping if this is a function definition we need to remove entirely
                if "def _fake_torch_device" in line:
                    skip_until_blank = True
                continue
            filtered_lines.append(line)
        patched = "\n".join(filtered_lines)
    return patched, changed


def _write_bundle(bundle: KernelBundle, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, content in bundle.files.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        paths[name] = str(path.resolve())
    return paths


def _compose_musa(
    problem_path: Path,
    subgraphs_path: Path,
    kernels_summary_path: Path,
    out_dir: Path,
    model_name: str,
    verify: bool,
    max_iters: int,
    target_platform: str,
) -> dict[str, Any]:
    if get_model_provider is None:
        raise SystemExit("KernelAgent providers unavailable")
    platform = get_platform(target_platform)
    backend = get_kernel_backend("musa")
    problem_code = _read_text(problem_path)
    subgraphs = json.loads(_read_text(subgraphs_path))
    if not isinstance(subgraphs, list):
        raise SystemExit("subgraphs.json must be a JSON array")
    kernels = _load_kernels_from_summary(kernels_summary_path, expected_backend="musa")
    provider = get_model_provider(model_name)
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    last_bundle: KernelBundle | None = None
    last_usage = None
    verify_info: dict[str, Any] = {}

    for attempt in range(1, max_iters + 1):
        if last_bundle is None:
            prompt = _build_musa_composition_prompt(
                problem_code, subgraphs, kernels, target_platform=platform
            )
        else:
            prompt = _build_musa_refinement_prompt(
                problem_code,
                subgraphs,
                last_bundle,
                {
                    "stdout": str(verify_info.get("stdout_tail", "")),
                    "stderr": str(verify_info.get("stderr_tail", "")),
                },
                target_platform=platform,
            )
        (attempts_dir / f"attempt_{attempt}.prompt.txt").write_text(
            prompt, encoding="utf-8"
        )
        response = provider.get_response(
            model_name, [{"role": "user", "content": prompt}], max_tokens=50000
        )
        last_usage = response.usage
        raw_text = response.content or ""
        bundle = extract_kernel_bundle(raw_text)
        if bundle is None:
            verify_info = {
                "verify_passed": False,
                "verify_reason": "MUSA response did not contain FILE blocks",
                "stderr_tail": "MUSA response did not contain FILE blocks",
                "stdout_tail": "",
            }
            continue
        try:
            bundle.require(backend)
        except ValueError as exc:
            verify_info = {
                "verify_passed": False,
                "verify_reason": str(exc),
                "stderr_tail": str(exc),
                "stdout_tail": "",
            }
            last_bundle = bundle
            continue

        attempt_dir = attempts_dir / f"attempt_{attempt}"
        _write_bundle(bundle, attempt_dir)
        last_bundle = bundle
        if verify:
            rr = run_bundle_candidate(
                bundle,
                run_root=out_dir / "runs",
                timeout_s=backend.verification_timeout_s,
                isolated=False,
                deny_network=False,
                entrypoint="kernel.py",
            )
            stdout_tail = ""
            stderr_tail = ""
            try:
                stdout_tail = rr.stdout_path.read_text(encoding="utf-8")[-4000:]
                stderr_tail = rr.stderr_path.read_text(encoding="utf-8")[-4000:]
            except Exception:
                pass
            verify_info = {
                "verify_rc": rr.rc,
                "verify_passed": rr.passed,
                "verify_reason": rr.reason,
                "validator": rr.validator_used,
                "stdout_path": str(rr.stdout_path),
                "stderr_path": str(rr.stderr_path),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            if rr.passed:
                break
        else:
            verify_info = {"verify_passed": True}
            break

    if last_bundle is None:
        raise SystemExit("MUSA composer did not produce a valid bundle")
    final_dir = out_dir / "composed_bundle"
    files = _write_bundle(last_bundle, final_dir)
    result: dict[str, Any] = {
        "success": bool(verify_info.get("verify_passed", not verify)),
        "composed_path": files.get("kernel.py"),
        "artifact_dir": str(final_dir.resolve()),
        "files": files,
        "model": model_name,
        "usage": last_usage,
        "rounds": attempt,
        "target_platform": target_platform,
        "kernel_backend": "musa",
    }
    result.update(verify_info)
    (out_dir / "composition_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return result


def compose(
    problem_path: Path,
    subgraphs_path: Path,
    kernels_summary_path: Path,
    out_dir: Path,
    model_name: str,
    verify: bool = False,
    max_iters: int = 5,
    target_platform: str = "cuda",
    kernel_backend: str = "triton",
) -> dict[str, Any]:
    if get_model_provider is None:
        raise SystemExit(
            "KernelAgent providers unavailable; ensure package import and dependencies"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    platform = get_platform(target_platform)
    backend = get_kernel_backend(kernel_backend)
    if not backend.supports_platform(platform.name):
        raise ValueError(
            f"Kernel backend {backend.name!r} does not support platform {platform.name!r}"
        )
    if backend.name == "musa":
        return _compose_musa(
            problem_path=problem_path,
            subgraphs_path=subgraphs_path,
            kernels_summary_path=kernels_summary_path,
            out_dir=out_dir,
            model_name=model_name,
            verify=verify,
            max_iters=max_iters,
            target_platform=target_platform,
        )
    provider = get_model_provider(model_name)

    # Platform
    # Load inputs
    problem_code = _read_text(problem_path)
    subgraphs = json.loads(_read_text(subgraphs_path))
    if not isinstance(subgraphs, list):
        raise SystemExit("subgraphs.json must be a JSON array")
    kernels = _load_kernels_from_summary(
        kernels_summary_path, expected_backend=backend.name
    )

    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    last_usage = None
    last_code = None
    verify_info: dict[str, Any] = {}

    for i in range(1, max_iters + 1):
        if i == 1 or last_code is None:
            prompt = _build_composition_prompt(
                problem_code, subgraphs, kernels, target_platform=platform
            )
        else:
            # Build refinement using previous error info
            stderr_tail = ""
            stdout_tail = ""
            try:
                if verify_info.get("stderr_path"):
                    with open(
                        verify_info["stderr_path"],
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        stderr_tail = f.read()[-2000:]
                if verify_info.get("stdout_path"):
                    with open(
                        verify_info["stdout_path"],
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        stdout_tail = f.read()[-2000:]
            except Exception:
                pass
            prompt = _build_refinement_prompt(
                problem_code,
                subgraphs,
                kernels,
                previous_code=last_code,
                error_info={"stderr_tail": stderr_tail, "stdout_tail": stdout_tail},
                target_platform=platform,
            )

        (attempts_dir / f"attempt_{i}.prompt.txt").write_text(prompt, encoding="utf-8")
        response = provider.get_response(
            model_name, [{"role": "user", "content": prompt}], max_tokens=50000
        )
        last_usage = response.usage
        raw_text = response.content or ""

        # Extract code
        extracted = extract_single_python_file(raw_text)
        code = extracted.code
        # Auto-patch trivial Triton pitfalls before running
        code, changed = _auto_patch_common_triton_issues(code, platform)
        (attempts_dir / f"attempt_{i}.py").write_text(code, encoding="utf-8")
        last_code = code

        # Verify each attempt if requested
        if verify:
            rr = run_candidate(
                artifacts_code_path=attempts_dir / f"attempt_{i}.py",
                run_root=out_dir / "runs",
                timeout_s=2400,
                isolated=False,
                deny_network=False,
            )
            verify_info = {
                "verify_rc": rr.rc,
                "verify_passed": rr.passed,
                "verify_reason": rr.reason,
                "validator": rr.validator_used,
                "stdout_path": str(rr.stdout_path),
                "stderr_path": str(rr.stderr_path),
            }
            if rr.passed:
                break
        else:
            # If not verifying, stop after first attempt
            break

    # Write final composed file as the last attempt
    composed_path = out_dir / "composed_kernel.py"
    composed_path.write_text(last_code or "", encoding="utf-8")

    result: dict[str, Any] = {
        "success": bool(verify_info.get("verify_passed", not verify)),
        "composed_path": str(composed_path.resolve()),
        "model": model_name,
        "usage": last_usage,
        "rounds": i,
        "target_platform": target_platform,
        "kernel_backend": backend.name,
    }
    result.update(verify_info)

    # Persist a small summary
    (out_dir / "composition_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Compose an end-to-end backend kernel from subgraphs + generated kernels"
    )
    p.add_argument(
        "--problem", required=True, help="Absolute path to KernelBench problem file"
    )
    p.add_argument(
        "--subgraphs", required=True, help="Path to subgraphs.json from Fuser"
    )
    p.add_argument(
        "--kernels-summary",
        required=True,
        help="Path to summary.json from dispatch step",
    )
    p.add_argument(
        "--out-dir",
        default="compose_out",
        help="Output directory for composed artifacts",
    )
    p.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL") or "gpt-5", help="LLM model name"
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Execute generated file and check PASS sentinel",
    )
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform (default: cuda)",
    )
    p.add_argument(
        "--kernel-backend",
        default="triton",
        choices=get_kernel_backend_choices(),
        help="Kernel source backend (default: triton)",
    )
    p.add_argument("--max-iters", type=int, default=5, help="Max LLM refinement rounds")
    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    subgraphs_path = Path(args.subgraphs).resolve()
    kernels_summary_path = Path(args.kernels_summary).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not problem_path.is_file():
        print(f"problem file not found: {problem_path}")
        return 2
    if not subgraphs_path.is_file():
        print(f"subgraphs file not found: {subgraphs_path}")
        return 2
    if not kernels_summary_path.is_file():
        print(f"kernels summary not found: {kernels_summary_path}")
        return 2

    try:
        res = compose(
            problem_path=problem_path,
            subgraphs_path=subgraphs_path,
            kernels_summary_path=kernels_summary_path,
            out_dir=out_dir,
            model_name=args.model,
            verify=args.verify,
            max_iters=args.max_iters,
            target_platform=args.target_platform,
            kernel_backend=args.kernel_backend,
        )
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        print(f"compose failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
