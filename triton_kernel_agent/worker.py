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

"""Verification Worker for testing and refining individual kernels."""

import ast
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.kernel_backend import (
    KernelBundle,
    extract_kernel_bundle,
    get_kernel_backend,
    validate_musa_bundle_semantics,
    validate_template_instantiations,
)
from triton_kernel_agent.worker_util import format_test_code_for_llm
from utils.providers import get_model_provider
from utils.providers.base import resolve_output_cap

from .prompt_manager import PromptManager
from .test_validation import validate_test_autograd, validate_test_layout, validate_kernel_globals
from .worker_util import _run_test_multiprocess


DISALLOWED_TORCH_PATTERNS = [
    (
        re.compile(r"\bimport\s+torch\.nn(\b|\s+as\b)"),
        "importing torch.nn modules is not allowed",
    ),
    (
        re.compile(r"\bfrom\s+torch\s+import\s+nn\b"),
        "importing torch.nn modules is not allowed",
    ),
    (
        re.compile(r"\bimport\s+torch\.nn\.functional\s+as\s+F\b"),
        "aliasing torch.nn.functional as F is not allowed",
    ),
    (re.compile(r"\btorch\.nn\."), "torch.nn module usage is not allowed"),
    (
        re.compile(r"\btorch\.nn\.functional\b"),
        "torch.nn.functional usage is not allowed",
    ),
    (
        re.compile(r"\bF\.[A-Za-z_]+\("),
        "torch.nn.functional alias calls (F.*) are not allowed",
    ),
    (re.compile(r"\btorch\.conv"), "torch convolution helpers are not allowed"),
    (
        re.compile(
            r"\btorch\.(relu|sigmoid|tanh|softmax|gelu|mish|hardtanh|max_pool|avg_pool)[A-Za-z0-9_]*\("
        ),
        "PyTorch activation/pooling helpers are not allowed",
    ),
    (
        re.compile(r"\bclass\s+\w+\s*\(\s*nn\.Module"),
        "Subclassing torch.nn.Module is not allowed",
    ),
    (
        re.compile(r"\.forward\("),
        "Calling .forward() indicates torch.nn module usage and is not allowed",
    ),
    (
        re.compile(r"\btorch\.ops\.aten\b"),
        "Low-level torch.ops.aten.* calls are not allowed; implement these ops directly in Triton kernels instead of relying on PyTorch compute",
    ),
    # Generic tensor-tensor math that must be implemented in Triton kernels
    (
        re.compile(r"\btorch\.(matmul|mm|bmm)\s*\("),
        "PyTorch matmul/mm/bmm tensor-tensor ops are not allowed; implement these in Triton kernels",
    ),
    (
        re.compile(r"\.(matmul|mm|bmm)\s*\("),
        "Tensor.matmul/mm/bmm methods are not allowed; implement these in Triton kernels",
    ),
    (
        re.compile(r"\btorch\.einsum\s*\("),
        "torch.einsum is not allowed; implement this contraction with Triton primitives",
    ),
    (
        re.compile(r"\.einsum\s*\("),
        "Tensor.einsum is not allowed; implement this contraction with Triton primitives",
    ),
    # Introspection / frame inspection that can be used to steal test locals
    (
        re.compile(r"\bimport\s+inspect\b"),
        "inspect-based reflection is not allowed inside kernel files",
    ),
    (
        re.compile(r"\binspect\.(stack|currentframe|getouterframes)\s*\("),
        "inspect stack/frame introspection is not allowed in kernels",
    ),
    (
        re.compile(r"\bsys\._getframe\s*\("),
        "sys._getframe is not allowed in kernels; do not access caller frames",
    ),
    (
        re.compile(r"\.f_locals\b|\.f_globals\b"),
        "Accessing frame locals/globals (f_locals/f_globals) from kernels is not allowed",
    ),
    (
        re.compile(r"\bglobals\s*\("),
        "globals() is not allowed in kernels; avoid depending on ambient test state",
    ),
    (
        re.compile(r"\blocals\s*\("),
        "locals() is not allowed in kernels; avoid depending on caller scopes",
    ),
]


# ``torch.musa`` is a backend attribute created by ``import torch_musa``; it is
# not an importable submodule, so ``import torch.musa`` / ``from torch.musa``
# raise ModuleNotFoundError in generated tests. Normalize those statements to the
# importable ``torch_musa`` package so the test also runs when executed directly.
_MUSA_IMPORT_PATTERNS = (
    (re.compile(r"^([ \t]*)import[ \t]+torch\.musa[ \t]*$", re.MULTILINE),
     r"\1import torch_musa"),
    (re.compile(r"^([ \t]*)import[ \t]+torch\.musa[ \t]+as[ \t]+([A-Za-z_]\w*)", re.MULTILINE),
     r"\1import torch_musa as \2"),
    (re.compile(r"^([ \t]*)from[ \t]+torch\.musa[ \t]+import[ \t]+", re.MULTILINE),
     r"\1from torch_musa import "),
)


def normalize_musa_imports(source: str) -> str:
    """Rewrite the invalid ``import torch.musa`` form to ``import torch_musa``."""
    for pattern, replacement in _MUSA_IMPORT_PATTERNS:
        source = pattern.sub(replacement, source)
    return source


# ---------------- Fix 6-1: output-cap truncation continuation ----------------
#
# The detector/continuation constants live in .llm_truncation so the agent seed
# path can reuse them (previously only the worker knew about truncation, so a
# seed cut at the cap was "repaired" by full regeneration under the same cap).
from .llm_truncation import (  # noqa: E402  (import after the pattern tables)
    CONTINUATION_PROMPT,
    LLMCallResult,
    MAX_CONTINUATIONS,
    TRUNCATION_MARGIN_TOKENS,
    looks_truncated as _looks_truncated,
)


# ---------------- Fix P0 (E): bound error text fed back into prompts ----------------
#
# A failing test can dump hundreds of KB to stdout/stderr; feeding that verbatim
# into the refinement prompt pushes the server over its prompt-token limit
# (observed 413: 176,943 tokens > 65,536). Keep the head (error type) and tail
# (real traceback/assert) and elide the middle. The full text stays in
# round_*.json for diagnosis.
_ERROR_CLIP_CHARS = 16000


def _clip_error_text(text: str, max_chars: int = _ERROR_CLIP_CHARS) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    head = max_chars // 2
    tail = max_chars - head
    return (
        f"{text[:head]}\n\n... [{len(text) - max_chars} chars elided] ...\n\n"
        f"{text[-tail:]}"
    )


# ---------------- Fix P1 (B): classify malformed responses for next-round backstop ---
#
# A malformed response is a failed *attempt*, not a fatal error (fix P1 B): it
# should consume one round and be retried, not terminate the whole run. But the
# recovery prompt must match what a kernel-understanding model can actually act
# on. We split the malformed reasons three ways:
#   ① content  : nothing usable was produced -> regenerate fresh (no format nudge).
#   ② format   : code exists but the FILE-block structure is wrong -> re-state the
#                format contract in the model's own instruction vocabulary (the raw
#                "missing required files: ..." diagnostic is a validation-side
#                symptom the model was never trained to interpret).
#   ③ semantic : operator/contract defect (toolchain, device builtins, ...) -> feed
#                the reason directly; it is in the model's comprehension domain.
_MALFORMED_CONTENT_MARKERS = (
    "no Python kernel code was extracted",
    "invalid Python syntax",
)
_MALFORMED_FORMAT_MARKERS = (
    "missing required files",
    "missing required top-level kernel_function",
    "kernel.py is missing required kernel_function",
)

_FORMAT_NUDGE = (
    "Repair only the complete bundle output format. "
    "Return all four files with literal FILE: kernel.py, FILE: binding.cpp, "
    "FILE: kernel.mu and FILE: setup.py headers immediately before fenced code. "
    "No prose, reference Model, input factories or test code in kernel.py. "
    "Preserve the public contract and do not weaken any test."
)


def _classify_malformed(reason: str) -> str:
    """Return 'content' | 'format' | 'semantic' for a malformed-reason string."""
    if any(m in reason for m in _MALFORMED_CONTENT_MARKERS):
        return "content"
    if any(m in reason for m in _MALFORMED_FORMAT_MARKERS):
        return "format"
    return "semantic"


class VerificationWorker:
    """Worker that verifies and refines a single kernel implementation."""

    def __init__(
        self,
        worker_id: int,
        workdir: Path,
        log_dir: Path,
        max_rounds: int = 10,
        history_size: int = 8,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-5",
        high_reasoning_effort: bool = True,
        target_platform: str = "cuda",
        no_cusolver: bool = False,
        test_timeout_s: int = 30,
        kernel_backend: str = "triton",
    ):
        """
        Initialize a verification worker.

        Args:
            worker_id: Unique identifier for this worker
            workdir: Working directory for this worker
            log_dir: Directory for logging
            max_rounds: Maximum refinement rounds
            history_size: Number of recent rounds to keep
            openai_api_key: OpenAI API key for refinement
            openai_model: Model name for refinement
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
            target_platform: Target platform default: cuda
            no_cusolver: If True, disables cuSolver library usage
            test_timeout_s: Timeout in seconds for test execution
        """
        self.worker_id = worker_id
        self.workdir = Path(workdir)
        self.log_dir = Path(log_dir)
        self.max_rounds = max_rounds
        self.history_size = history_size
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self._platform_config = get_platform(target_platform)
        self.no_cusolver = no_cusolver
        self.test_timeout_s = test_timeout_s
        self._kernel_backend = get_kernel_backend(kernel_backend)

        # Setup files
        self.kernel_file = self.workdir / "kernel.py"
        self.test_files: list[Path] = []

        # History for LLM context
        self.history = deque(maxlen=history_size)

        # Setup logging early so it is available for any error paths
        self._setup_logging()

        # Initialize prompt manager with resolved config
        self.prompt_manager = PromptManager(
            target_platform=self._platform_config,
            kernel_backend=self._kernel_backend,
        )

        # Initialize provider (may be unavailable in offline/test environments)
        self.provider = None
        try:
            self.provider = get_model_provider(self.openai_model)
        except ValueError as e:
            # Provider not available, will use mock mode
            self.logger.warning(f"Provider not available: {e}")

        # Fix 6-1: resolve the model's actual output cap once so the
        # truncation threshold and requested max_tokens track a model swap
        # instead of a hardcoded 16000/20000 pair.
        self.output_cap = resolve_output_cap(self.openai_model, self.provider)
        self._truncation_threshold = self.output_cap - TRUNCATION_MARGIN_TOKENS
        self.logger.info(
            "Output cap resolved for %s: %d (truncation threshold %d; set "
            "KERNELAGENT_OUTPUT_CAP or extend the provider registry if the "
            "model supports more)",
            self.openai_model,
            self.output_cap,
            self._truncation_threshold,
        )

        # Fix P1 (F): whole-run wall-clock budget. Once a malformed response no
        # longer terminates the run (fix P1 B), every round can legitimately be
        # consumed, so the round cap alone can no longer bound wall time. This
        # is the explicit backstop; 0 disables it.
        self.round_budget_s = int(os.getenv("KERNELAGENT_ROUND_BUDGET_S", "10800"))

    def _setup_logging(self):
        """Setup worker-specific logging."""
        log_file = self.log_dir / f"worker_{self.worker_id}.log"
        self.logger = logging.getLogger(f"worker_{self.worker_id}")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def _extract_code_from_response(
        self,
        response_text: str,
        language: str = "python",
        prefer_kernel_function: bool = False,
    ) -> str | None:
        """
        Extract code from LLM response text.

        Args:
            response_text: The full LLM response text
            language: The expected language (default: python)
            prefer_kernel_function: When True and multiple code blocks are
                found, prefer the block that defines ``kernel_function``
                (falling back to the longest block).  Use this when the
                prompt contains additional test code that the LLM may echo
                back.

        Returns:
            Extracted code or None if no valid code block found
        """
        if not response_text:
            return None

        # First, try to find code blocks with language markers
        # Pattern matches ```python or ```language_name
        pattern = rf"```{language}\s*\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if not matches:
            # Try generic code blocks without language marker
            pattern = r"```\s*\n(.*?)```"
            matches = re.findall(pattern, response_text, re.DOTALL)

        if matches:
            if prefer_kernel_function and len(matches) > 1:
                # When additional tests are in the prompt the LLM may echo
                # wrapper code.  Prefer the block defining kernel_function.
                for block in matches:
                    if re.search(r"\bdef\s+kernel_function\b", block):
                        return block.strip()
                # Fallback: return the longest block
                return max(matches, key=len).strip()
            # Default: return the first match
            return matches[0].strip()

        # If no code blocks found, check if the entire response looks like code
        # This is a fallback for cases where LLM doesn't use code blocks
        lines = response_text.strip().split("\n")

        # Simple heuristic: if response contains import statements or function definitions
        code_indicators = ["import ", "from ", "def ", "class ", "@", '"""', "'''"]
        if any(
            line.strip().startswith(indicator)
            for line in lines
            for indicator in code_indicators
        ):
            # Likely the entire response is code
            return response_text.strip()

        # No code found
        self.logger.warning("No code block found in LLM response")
        return None

    def _validate_kernel_candidate(
        self, kernel_code: str | KernelBundle | None
    ) -> str | None:
        """Return a reason when extracted kernel code is structurally malformed."""
        if isinstance(kernel_code, str) and self._kernel_backend.name == "musa":
            kernel_code = extract_kernel_bundle(kernel_code)
        if isinstance(kernel_code, KernelBundle):
            try:
                kernel_code.require(self._kernel_backend)
            except ValueError as exc:
                return str(exc)
            wrapper = kernel_code.files.get("kernel.py", "")
            if "def kernel_function" not in wrapper:
                return "kernel.py is missing required kernel_function definition"
            reason = validate_musa_bundle_semantics(kernel_code)
            if reason:
                return reason
            try:
                for name, source in kernel_code.files.items():
                    if name.endswith(".py"):
                        validate_kernel_globals(source)
            except (ValueError, SyntaxError) as exc:
                return str(exc)
            return None
        if not kernel_code or not kernel_code.strip():
            return "no Python kernel code was extracted from the model response"

        try:
            module = ast.parse(kernel_code)
        except SyntaxError as exc:
            line = f"line {exc.lineno}" if exc.lineno else "unknown line"
            return f"invalid Python syntax ({line}: {exc.msg})"

        has_kernel_function = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "kernel_function"
            for node in module.body
        )
        if not has_kernel_function:
            return "missing required top-level kernel_function definition"

        try:
            validate_kernel_globals(kernel_code)
        except ValueError as exc:
            return str(exc)

        return None

    def _test_validation_error(self, tests: list[str]) -> str | None:
        if self._kernel_backend.name != "musa":
            return None
        for index, source in enumerate(tests):
            try:
                validate_test_autograd(source)
                validate_test_layout(source)
            except ValueError as exc:
                return f"Invalid correctness test [{index}]; kernel refinement skipped: {exc}"
        return None

    def _write_kernel(self, kernel_code: str | KernelBundle):
        """Write only the kernel code to file."""
        if isinstance(kernel_code, str) and self._kernel_backend.name == "musa":
            parsed = extract_kernel_bundle(kernel_code)
            if parsed is not None:
                kernel_code = parsed
        if isinstance(kernel_code, KernelBundle):
            for name, content in kernel_code.files.items():
                (self.workdir / name).write_text(content, encoding="utf-8")
        else:
            self.kernel_file.write_text(kernel_code, encoding="utf-8")
        self.logger.info("Updated kernel file")

    def _write_files(self, kernel_code: str | KernelBundle, test_code: list[str]):
        """Write kernel and test code to files.

        Note: The test code should import the kernel function from the kernel file:
            from kernel import kernel_function

        Both files are written to the same directory (workdir).

        Args:
            kernel_code: The kernel source code.
            test_code: List of test code strings. ``test_code[0]`` is the
                primary test written to ``test_kernel.py``; any subsequent
                entries are written to ``test_extra_{i}_kernel.py``.
        """
        self._write_kernel(kernel_code)
        from . import case_suite, kernel_call_binding
        shutil.copy2(case_suite.__file__, self.workdir / "kernel_case_suite.py")
        shutil.copy2(kernel_call_binding.__file__, self.workdir / "kernel_call_binding.py")
        self.test_files = []
        is_musa = self._kernel_backend.name == "musa"
        for i, code in enumerate(test_code):
            name = "test_kernel.py" if i == 0 else f"test_extra_{i}_kernel.py"
            path = self.workdir / name
            if is_musa:
                code = normalize_musa_imports(code)
            path.write_text(code)
            self.test_files.append(path)
        from .case_suite import SUITE_MARKER
        self.test_files.sort(key=lambda path: SUITE_MARKER not in path.read_text())
        self.logger.info("Wrote kernel and %d test file(s)", len(self.test_files))

    def _strip_comments_and_strings(self, code: str) -> str:
        """Remove comments and docstrings to avoid false positives when scanning code."""
        pattern = re.compile(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#.*)')
        return re.sub(pattern, "", code)

    def _detect_pytorch_compute(self, kernel_code: str | KernelBundle) -> str | None:
        """Detect disallowed PyTorch usage inside the kernel wrapper."""
        if isinstance(kernel_code, str) and self._kernel_backend.name == "musa":
            parsed = extract_kernel_bundle(kernel_code)
            if parsed is not None:
                kernel_code = parsed
        source = (
            kernel_code.files.get("kernel.py", "")
            if isinstance(kernel_code, KernelBundle)
            else kernel_code
        )
        sanitized = self._strip_comments_and_strings(source)
        for pattern, message in DISALLOWED_TORCH_PATTERNS:
            # Native MUSA's Python wrapper legitimately calls the compiled
            # extension entry point (commonly ``extension.forward(...)``).
            # The remaining torch.nn and tensor-compute bans still prevent a
            # PyTorch fallback from bypassing native kernel execution.
            if self._kernel_backend.name == "musa" and pattern.pattern == r"\.forward\(":
                continue
            if pattern.search(sanitized):
                return message
        return None

    @staticmethod
    def _failure_diagnostic(result: subprocess.CompletedProcess) -> str:
        """Return the most actionable fragment of a failed test's real output.

        When a suite companion test crashes before emitting its ``RESULT_PREFIX``
        line (or emits no usable rows), the underlying process error (e.g. an
        ImportError, ModuleNotFoundError, or a compiler traceback) is far more
        actionable than a fixed placeholder string. Prefer the tail of stderr
        (which carries Python/toolchain tracebacks), falling back to stdout, and
        keep the fragment bounded so it fits into the refinement prompt.
        """
        fragment = (result.stderr or "").strip()
        if not fragment:
            fragment = (result.stdout or "").strip()
        if not fragment:
            return ""
        return fragment[-2000:]

    def _reference_requires_chunking(self) -> bool:
        """True when the explicit case suite had to evaluate its reference in chunks.

        Chunking only happens when the reference output exceeds the MUSA int32 element
        limit (2**31); a full-shape reference there silently returns zeros, so a
        generated test comparing against a full-shape reference cannot be authoritative.
        """
        return any(row.get("reference_chunked") for row in self.last_case_results)

    def _run_test(self) -> tuple[bool, str, str]:
        """
        Run the explicit companion first, then remaining scripts sequentially.

        Returns:
            Tuple of (success, stdout, stderr)
        """
        self.last_case_results = []
        try:
            if self._kernel_backend.name == "musa":
                build = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=str(self.workdir),
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout_s,
                )
                if build.returncode != 0:
                    build_diagnostics = "\n".join(
                        part for part in (build.stdout, build.stderr) if part
                    )
                    self.logger.error(
                        "Native MUSA extension build failed. Exit code: %s, diagnostics: %s",
                        build.returncode,
                        build_diagnostics[-12000:],
                    )
                    return False, build.stdout, build_diagnostics
                self.logger.info("Native MUSA extension build passed")

            for test_file in self.test_files:
                if not test_file.exists():
                    return False, "", f"Required test file missing: {test_file.name}"
                command = [sys.executable, str(test_file)]
                if self._kernel_backend.name == "musa":
                    command = [sys.executable, "-c",
                               "from kernel_case_suite import run_guarded_test; "
                               "import sys; run_guarded_test(sys.argv[1], 'musa')", str(test_file)]
                result = subprocess.run(
                    command,
                    cwd=str(self.workdir),
                    capture_output=True,
                    text=True,
                    timeout=self.test_timeout_s,
                )
                from .case_suite import RESULT_PREFIX
                for line in result.stdout.splitlines():
                    if line.startswith(RESULT_PREFIX):
                        rows = json.loads(line[len(RESULT_PREFIX):])
                        if not isinstance(rows, list) or not rows:
                            return False, result.stdout, self._failure_diagnostic(result) or (
                                "Explicit case report is empty or invalid")
                        self.last_case_results = rows
                        self.logger.info("Explicit case coverage: %s", [(row["case_key"], row["status"]) for row in rows])
                        if any(row["status"] != "passed" for row in rows):
                            return False, result.stdout, self._failure_diagnostic(result) or (
                                "An explicit test case failed or was missing")
                from .case_suite import SUITE_MARKER
                if SUITE_MARKER in test_file.read_text() and not self.last_case_results:
                    return False, result.stdout, self._failure_diagnostic(result) or (
                        "Explicit case suite did not report coverage")
                if result.returncode != 0:
                    self.logger.error(
                        "Test %s failed. Exit code: %s, stdout: %s, stderr: %s",
                        test_file.name,
                        result.returncode,
                        result.stdout[:2000],
                        result.stderr[:2000],
                    )
                    return False, result.stdout, result.stderr
                self.logger.info("Test %s passed", test_file.name)
                if self._reference_requires_chunking():
                    self.logger.info(
                        "Skipping remaining generated tests for %s: the explicit case suite passed with a "
                        "chunked reference (an output of >= 2**31 elements is unsupported as a full-shape "
                        "MUSA reference), so a full-shape comparison in a generated test is not authoritative",
                        test_file.name,
                    )
                    return True, result.stdout, result.stderr

            return True, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            self.logger.error("Test timed out")
            return (
                False,
                "",
                f"Test execution timed out after {self.test_timeout_s} seconds",
            )
        except Exception as e:
            self.logger.error(f"Test execution error: {e}")
            return False, "", str(e)

    def _call_llm(self, messages: list, **kwargs) -> str:
        """
        Call the LLM provider for the configured model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text
        """
        return self._call_llm_detailed(messages, **kwargs).content

    def _call_llm_detailed(self, messages: list, **kwargs) -> LLMCallResult:
        """Like _call_llm but also surfaces completion_tokens and finish_reason.

        Both feed the truncation detector (fix 6-1): ``finish_reason ==
        'length'`` is the universal cap-hit signal on well-behaved servers,
        while the token threshold covers servers that report ``'stop'`` at
        the cap (all 10 truncated MusaCoder refinement responses in the
        P45-56 batch did exactly that).
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.openai_model}")

        # Add high_reasoning_effort to kwargs if set
        if self.high_reasoning_effort:
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.openai_model, messages, **kwargs)
        completion_tokens = None
        if isinstance(response.usage, dict):
            raw = response.usage.get("completion_tokens")
            if isinstance(raw, int):
                completion_tokens = raw
        return LLMCallResult(
            content=response.content,
            completion_tokens=completion_tokens,
            finish_reason=getattr(response, "finish_reason", None),
        )

    def _refine_kernel(
        self,
        kernel_code: str | KernelBundle,
        error_info: dict[str, str],
        problem_description: str,
        test_code: str,
    ) -> str | KernelBundle:
        """
        Refine kernel based on error information using OpenAI API.

        Uses multi-turn dialogue by incorporating history of previous attempts.
        """
        if self.provider:
            try:
                self.logger.info(f"Refining kernel using {self.openai_model}")

                # Build context from history
                history_context = ""
                if self.history:
                    history_context = "\n\nPREVIOUS ATTEMPTS:\n"
                    for i, round_data in enumerate(self.history):
                        history_context += f"\nAttempt {i + 1}:\n"
                        prior = round_data["kernel_code"]
                        prior_text = (
                            KernelBundle(prior).render_for_prompt()
                            if isinstance(prior, dict)
                            else prior
                        )
                        history_context += f"Kernel code:\n{prior_text[:500]}...\n"
                        if round_data.get("stderr"):
                            history_context += f"Error: {round_data['stderr'][:2000]}\n"
                        if round_data.get("stdout"):
                            history_context += (
                                f"Output: {round_data['stdout'][:1000]}\n"
                            )

                # Create refinement prompt using template
                prompt = self.prompt_manager.render_kernel_refinement_prompt(
                    problem_description=problem_description,
                    test_code=test_code,
                    kernel_code=(
                        kernel_code.render_for_prompt()
                        if isinstance(kernel_code, KernelBundle)
                        else kernel_code
                    ),
                    error_info=error_info,
                    history_context=history_context,
                    no_cusolver=self.no_cusolver,
                )

                # Call LLM API — truncation continuation (fix 6-1):
                #   A truncated response is continued on the extended thread
                #   (<= MAX_CONTINUATIONS). Fix P1 (A) removed the same-round
                #   Phase-2 format repair: a still-malformed response is raised
                #   so run()'s backstop (fix P1 B) consumes a round and
                #   regenerates next round.
                base_messages = [{"role": "user", "content": prompt}]
                thread = list(base_messages)
                accumulated = ""
                segment = self._call_llm_detailed(thread, max_tokens=self.output_cap)
                refined_kernel = None
                malformed_reason = None
                last_segment = ""
                last_reason = "no response"
                for cont in range(MAX_CONTINUATIONS + 1):
                    candidate = accumulated + segment.content
                    try:
                        refined_kernel = (
                            extract_kernel_bundle(candidate)
                            if self._kernel_backend.name == "musa"
                            else self._extract_code_from_response(
                                candidate,
                                prefer_kernel_function=getattr(self, "_has_multiple_tests", False),
                            )
                        )
                        malformed_reason = self._validate_kernel_candidate(refined_kernel)
                    except ValueError as exc:
                        refined_kernel = None
                        malformed_reason = str(exc)
                    if not malformed_reason:
                        break
                    last_segment, last_reason = segment.content, malformed_reason
                    if cont == MAX_CONTINUATIONS or not _looks_truncated(
                        segment, malformed_reason, self._truncation_threshold
                    ):
                        break
                    # Preserve the raw truncated segment for diagnosis.
                    (self.log_dir / f"refinement_truncated_{len(self.history)}_{cont}.txt").write_text(
                        segment.content, encoding="utf-8")
                    self.logger.warning(
                        "Truncated MUSA refinement (cont=%d, completion_tokens=%s, "
                        "finish_reason=%s); requesting continuation",
                        cont, segment.completion_tokens, segment.finish_reason,
                    )
                    thread.extend([
                        {"role": "assistant", "content": segment.content},
                        {"role": "user", "content": CONTINUATION_PROMPT},
                    ])
                    accumulated = candidate
                    segment = self._call_llm_detailed(thread, max_tokens=self.output_cap)

                # A partial native bundle that failed validation (e.g. only
                # kernel.py survived the truncation) must not be returned as a
                # success. Fix P1 (A): there is no same-round format repair —
                # raise so run()'s backstop (fix P1 B) consumes a round and
                # regenerates next round. Raw (non-native) backends keep the
                # legacy behavior of handing the candidate to the compile/test
                # round, which reports the real error back into the next
                # refinement.
                if refined_kernel and (
                    not malformed_reason or not isinstance(refined_kernel, KernelBundle)
                ):
                    self.logger.info(
                        f"Successfully refined kernel using {self.openai_model}"
                    )
                    return refined_kernel

                # Preserve the rejected segment for diagnosis, then raise. The
                # round loop (run()) turns this into a next-round regeneration
                # instead of terminating the whole run (fix P1 B).
                (self.log_dir / f"refinement_rejected_{len(self.history)}_0.txt").write_text(
                    last_segment, encoding="utf-8")
                raise ValueError(f"Malformed LLM kernel response: {last_reason}")

            except Exception as e:
                if "Malformed LLM kernel response" in str(e):
                    raise
                self.logger.error(f"Error refining kernel with LLM API: {e}")
                # Fall back to mock refinement

        # Mock refinement (fallback)
        self.logger.info("Refining kernel (mock implementation)")

        # For testing, make a simple modification
        if "error" in error_info.get("stderr", "").lower():
            # Add a comment to show refinement happened
            if isinstance(kernel_code, str):
                return f"# Refinement attempt {len(self.history) + 1}\n{kernel_code}"

        return kernel_code

    def _log_round(
        self,
        round_num: int,
        success: bool,
        kernel_code: str | KernelBundle,
        stdout: str,
        stderr: str,
    ):
        """Log the results of a verification round."""
        round_data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "kernel_code": (
                dict(kernel_code.files)
                if isinstance(kernel_code, KernelBundle)
                else kernel_code
            ),
            "stdout": stdout,
            "stderr": stderr,
        }

        # Save to log file
        round_log_file = self.log_dir / f"round_{round_num}.json"
        with open(round_log_file, "w") as f:
            json.dump(round_data, f, indent=2)

        # Add to history
        self.history.append(round_data)

    def _failed_run(self, rounds: int, **details) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "success": False, "rounds": rounds,
                "history": list(self.history), **details}

    def _malformed_error_info(
        self, reason: str, last_stdout: str, last_stderr: str
    ) -> dict[str, Any]:
        """Build error_info for a round recovering from a malformed LLM response.

        The instruction is chosen by failure class (fix P1 B) so a
        kernel-understanding model is never asked to interpret a raw format
        diagnostic ("missing required files: ...") it was not trained on. The
        previous real test output is carried along so the model can fix both
        format and semantics in the same regeneration.
        """
        kind = _classify_malformed(reason)
        if kind == "content":
            nudge = (
                "The previous generation produced no extractable kernel code. "
                "Produce a fresh, complete implementation."
            )
        elif kind == "format":
            nudge = _FORMAT_NUDGE
        else:  # semantic: operator/contract defect, in the model's domain
            nudge = reason
        stderr = _clip_error_text(last_stderr)
        stderr = f"{stderr}\n\n{nudge}" if stderr else nudge
        return {
            "stdout": _clip_error_text(last_stdout),
            "stderr": stderr,
            "history": list(self.history),
        }

    def run(
        self,
        kernel_code: str | KernelBundle,
        test_code: list[str],
        problem_description: str,
        success_event: mp.Event,
    ) -> dict[str, Any]:
        """
        Run verification and refinement loop.

        Args:
            kernel_code: Initial kernel implementation
            test_code: List of test code strings (primary + additional tests)
            problem_description: Problem description for context
            success_event: Shared event to check if another worker succeeded

        Returns:
            Dictionary with results
        """
        self.logger.info(f"Starting verification for worker {self.worker_id}")
        test_error = self._test_validation_error(test_code)
        if test_error:
            self.logger.error(test_error)
            return self._failed_run(0, error=test_error, failure_kind="invalid_correctness_test")
        self._has_multiple_tests = len(test_code) > 1

        t_start = time.monotonic()
        current_kernel = kernel_code
        # Fix P1 (B): a malformed response no longer terminates the run. The
        # reason is carried across rounds and consumed as one attempt each.
        pending_malformed: str | None = None
        last_stdout = last_stderr = ""

        malformed_reason = self._validate_kernel_candidate(current_kernel)
        if malformed_reason:
            pending_malformed = f"Malformed LLM kernel response: {malformed_reason}"
            self.logger.warning(
                f"❌ seed {pending_malformed}; round 1 will regenerate from it"
            )

        for round_num in range(self.max_rounds):
            # Check if another worker has succeeded
            if success_event.is_set():
                self.logger.info("Another worker succeeded, stopping")
                return {
                    "worker_id": self.worker_id,
                    "success": False,
                    "stopped_early": True,
                    "rounds": round_num,
                }

            # Fix P1 (F): explicit whole-run wall-clock budget. With malformed
            # responses now consuming rounds instead of terminating, the round
            # cap alone can no longer bound wall time.
            if self.round_budget_s and time.monotonic() - t_start > self.round_budget_s:
                self.logger.warning(
                    "Run budget exceeded after %.0f s (limit %d s)",
                    time.monotonic() - t_start, self.round_budget_s,
                )
                return self._failed_run(round_num, budget_exceeded=True)

            if pending_malformed is None:
                self.logger.info(f"Round {round_num + 1}/{self.max_rounds}")

                # Write files - test only on first round, kernel every round
                if round_num == 0:
                    # First round: write both kernel and test(s)
                    self._write_files(current_kernel, test_code)
                else:
                    # Subsequent rounds: only update kernel, test remains unchanged
                    self._write_kernel(current_kernel)

                # Run verification (additional tests chained automatically by _run_test)
                success, stdout, stderr, violation = self._single_verification_pass(
                    current_kernel
                )

                if violation:
                    success, stdout, stderr = False, "", violation

                # Log round
                self._log_round(round_num + 1, success, current_kernel, stdout, stderr)

                if success:
                    self.logger.info(
                        f"Success! Kernel passed test in round {round_num + 1}"
                    )
                    return {
                        "worker_id": self.worker_id,
                        "success": True,
                        "case_results": getattr(self, "last_case_results", []),
                        "kernel_code": (
                            dict(current_kernel.files)
                            if isinstance(current_kernel, KernelBundle)
                            else current_kernel
                        ),
                        "rounds": round_num + 1,
                        "history": list(self.history),
                    }

                last_stdout, last_stderr = stdout, stderr
                # Fix P0 (E): bound error text so a runaway stdout/stderr cannot
                # push the next refinement prompt over the server prompt limit.
                error_info = {
                    "stdout": _clip_error_text(stdout),
                    "stderr": _clip_error_text(stderr),
                    "history": list(self.history),
                }
            else:
                # Fix P1 (B): malformed recovery round. The previous kernel was
                # not regenerated successfully, so skip verification and
                # regenerate with a prompt tailored to the malformed class.
                self.logger.info(
                    f"Round {round_num + 1}/{self.max_rounds} (malformed recovery)"
                )
                self._log_round(round_num + 1, False, current_kernel, "", pending_malformed)
                error_info = self._malformed_error_info(
                    pending_malformed, last_stdout, last_stderr
                )

            try:
                current_kernel = self._refine_kernel(
                    current_kernel,
                    error_info,
                    problem_description,
                    format_test_code_for_llm(test_code),
                )
                malformed_reason = self._validate_kernel_candidate(current_kernel)
                if malformed_reason:
                    raise ValueError(f"Malformed LLM kernel response: {malformed_reason}")
            except ValueError as exc:
                if "Malformed LLM kernel response" not in str(exc):
                    raise
                pending_malformed = str(exc)
                self.logger.warning(f"❌ {pending_malformed}")
                continue
            pending_malformed = None

        # Max rounds reached without success
        self.logger.warning(f"Max rounds ({self.max_rounds}) reached without success")
        return self._failed_run(self.max_rounds, max_rounds_reached=True)

    def _single_verification_pass(
        self, kernel_code: str | KernelBundle
    ) -> tuple[bool, str, str, str | None]:
        """
        Run a single verification pass on the kernel.

        Returns:
            Tuple of (success, stdout, stderr, violation_message)
            - violation_message is set if PyTorch usage detected, None otherwise
        """
        self.last_case_results = []
        violation = self._detect_pytorch_compute(kernel_code)
        if violation:
            message = f"Disallowed PyTorch usage detected: {violation}"
            self.logger.error(message)
            return False, "", message, message

        if self._kernel_backend.name == "musa":
            reason = validate_template_instantiations(kernel_code)
            if reason:
                self.logger.error(reason)
                return False, "", reason, None

        from .case_suite import SUITE_MARKER
        has_suite = any(SUITE_MARKER in path.read_text() for path in self.test_files if path.exists())
        success, stdout, stderr = (
            self._run_test()
            if has_suite or os.getenv("KA_PROCESS_USE_SYS_EXECUTABLE", "1") == "1"
            else _run_test_multiprocess(
                self.logger,
                self.workdir,
                self.test_files,
            )
        )

        return success, stdout, stderr, None

    def verify_with_refinement(
        self,
        kernel_code: str | KernelBundle,
        test_code: list[str],
        problem_description: str,
        max_refine_attempts: int = 3,
    ) -> tuple[bool, str | KernelBundle, str]:
        """
        Verify kernel correctness with refinement attempts.

        This is a simpler API for single-pass verification with refinement,
        useful for optimization loops that manage their own iteration.

        Args:
            kernel_code: Kernel code to verify
            test_code: List of test code strings (primary + additional tests)
            problem_description: Problem description for refinement context
            max_refine_attempts: Maximum refinement attempts if verification fails

        Returns:
            Tuple of (success, final_kernel_code, error_feedback)
            - success: Whether the kernel passed verification
            - final_kernel_code: The verified (possibly refined) kernel
            - error_feedback: Error message if failed, empty string if success
        """
        current_kernel = kernel_code
        self._has_multiple_tests = len(test_code) > 1
        self.last_case_results = []

        test_error = self._test_validation_error(test_code)
        if test_error:
            self.logger.error(test_error)
            return False, current_kernel, test_error

        malformed_reason = self._validate_kernel_candidate(current_kernel)
        if malformed_reason:
            error_feedback = f"Malformed LLM kernel response: {malformed_reason}"
            self.logger.warning(f"❌ {error_feedback}")
            return False, current_kernel, error_feedback

        # Write files for testing (primary + additional tests)
        self._write_files(current_kernel, test_code)

        # Initial verification (additional tests chained automatically by _run_test)
        success, stdout, stderr, violation = self._single_verification_pass(
            current_kernel
        )

        if violation:
            # Log initial failure so refinement LLM sees it in history
            self._log_round(0, False, current_kernel, stdout, stderr)
            return False, current_kernel, violation

        if success:
            self.logger.info("✅ Verification passed on first attempt")
            return True, current_kernel, ""

        # Refinement loop
        for attempt in range(1, max_refine_attempts + 1):
            error_output = stderr if stderr.strip() else stdout
            self.logger.info(f"Refinement attempt {attempt}/{max_refine_attempts}...")

            error_info = {
                "stdout": _clip_error_text(stdout),
                "stderr": _clip_error_text(stderr),
                "error_type": (
                    "compilation"
                    if "CompilationError" in error_output
                    or "SyntaxError" in error_output
                    else "runtime"
                ),
            }

            # Refine kernel
            try:
                refined_kernel = self._refine_kernel(
                    current_kernel,
                    error_info,
                    problem_description,
                    format_test_code_for_llm(test_code),
                )
            except ValueError as exc:
                if "Malformed LLM kernel response" not in str(exc):
                    raise
                error_feedback = str(exc)
                self.logger.warning(f"❌ {error_feedback}")
                return False, current_kernel, error_feedback
            malformed_reason = self._validate_kernel_candidate(refined_kernel)
            if malformed_reason:
                error_feedback = f"Malformed LLM kernel response: {malformed_reason}"
                self.logger.warning(f"❌ {error_feedback}")
                return False, current_kernel, error_feedback

            # Write and test refined kernel
            self._write_kernel(refined_kernel)
            success, stdout, stderr, violation = self._single_verification_pass(
                refined_kernel
            )

            if violation:
                current_kernel = refined_kernel
                continue

            if success:
                self.logger.info(
                    f"✅ Verification passed after refinement (attempt {attempt})"
                )
                return True, refined_kernel, ""

            current_kernel = refined_kernel

        # All attempts exhausted
        error_output = stderr if stderr.strip() else stdout
        error_feedback = f"Verification failed after {max_refine_attempts} refinement attempts:\n{error_output[:2000]}"
        self.logger.warning(f"❌ {error_feedback[:200]}...")
        return False, current_kernel, error_feedback
