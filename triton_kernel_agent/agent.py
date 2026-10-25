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

"""Main Triton Kernel Generation Agent."""

import ast
import hashlib
import os
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime
import logging
from dotenv import load_dotenv

from .manager import WorkerManager
from .experience.generation import generation_memory
from .prompt_manager import PromptManager
from .test_validation import (
    validate_forward_only_test,
    validate_test_autograd,
    validate_test_kernel_call_binding,
    validate_test_layout,
)
from .verification_policy import is_forward_only, resolve_verify_direction
from .experience.signature import declared_execution_direction as _declared_execution_direction
from .llm_truncation import (
    CONTINUATION_PROMPT,
    LLMCallResult,
    MAX_CONTINUATIONS,
    TRUNCATION_MARGIN_TOKENS,
    looks_truncated,
)
from utils.providers import BaseProvider, get_model_provider
from utils.providers.base import resolve_output_cap
from triton_kernel_agent.platform_config import PlatformConfig, get_platform
from triton_kernel_agent.worker_util import format_test_code_for_llm
from triton_kernel_agent.kernel_backend import (
    KernelBundle,
    extract_kernel_bundle,
    get_kernel_backend,
    validate_musa_bundle_semantics,
)
from triton_kernel_agent.kernel_call_binding import kernel_signature_bindability
from triton_kernel_agent.experience import (
    ExperienceRecord,
    SQLiteExperienceStore,
    build_operator_signature,
    reference_source_from_test,
    collect_environment,
    format_verified_seed_context,
)

# How many complete-format repair requests to spend when a MUSA seed came back
# malformed. A single repair is not enough for verbose reasoning models that
# exhaust the completion budget mid-thought, because the repair prompt is longer
# than the seed prompt and tends to truncate again.
MUSA_SEED_REPAIR_ATTEMPTS = 3


def _parse_reference_problem(problem_description: str):
    """Parse the final Describe2 produced for this request, if present."""
    headings = list(re.finditer(r"(?m)^## Describe2[^\n]*\n", problem_description))
    source = problem_description[headings[-1].end():] if headings else problem_description
    if headings:
        source = re.split(r"(?m)^##\s+", source, maxsplit=1)[0].strip()
        fenced = re.fullmatch(r"```(?:python)?\s*\n([\s\S]*?)\n```", source)
        if fenced:
            source = fenced.group(1)
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _input_factory_contract(tree: ast.Module, name: str) -> dict[str, str]:

    """Capture a factory and its top-level helpers/constants, not just its body."""
    definitions: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        definitions[child.id] = node
    if name not in definitions:
        return {}
    selected = {name}
    pending = [name]
    while pending:
        for child in ast.walk(definitions[pending.pop()]):
            if (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                    and child.id in definitions and child.id not in selected):
                selected.add(child.id)
                pending.append(child.id)
    return {
        item: ast.dump(definitions[item], include_attributes=False)
        for item in selected
    }


def validate_generated_test_input_contract(
    problem_description: str, test_code: str
) -> None:
    """Ensure a standalone generated test reuses the exact input factories."""
    reference_tree = _parse_reference_problem(problem_description)
    if reference_tree is None:
        return
    reference_functions = {
        node.name: node
        for node in reference_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("get_inputs", "get_init_inputs")
    }
    if not reference_functions:
        return
    try:
        test_tree = ast.parse(test_code)
    except SyntaxError as exc:
        raise ValueError(
            f"Generated correctness test is not valid Python: {exc}"
        ) from exc
    test_functions = {
        node.name: node
        for node in test_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        node.func.id
        for node in ast.walk(test_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    errors = []
    for name, reference_node in reference_functions.items():
        test_node = test_functions.get(name)
        if test_node is None:
            errors.append(f"{name}() definition")
        elif (_input_factory_contract(test_tree, name)
              != _input_factory_contract(reference_tree, name)):
            errors.append(f"exact {name}() definition")
        if name not in calls:
            errors.append(f"{name}() call")
    if errors:
        raise ValueError(
            "Generated correctness test does not reuse the problem input contract; missing or changed "
            + ", ".join(errors)
        )


class TritonKernelAgent:
    """Main agent for generating and optimizing Triton kernels."""

    def __init__(
        self,
        num_workers: int | None = None,
        max_rounds: int | None = None,
        log_dir: str | None = None,
        model_name: str | None = None,
        high_reasoning_effort: bool = True,
        preferred_provider: BaseProvider | None = None,
        target_platform: PlatformConfig | None = None,
        no_cusolver: bool = False,
        test_timeout_s: int = 30,
        kernel_backend: str = "triton",
        experience_db_path: str | Path | None = None,
        enable_experience_memory: bool = True,
    ):
        """
        Initialize the Triton Kernel Agent.

        Args:
            num_workers: Number of parallel workers for verification (loaded from .env if None)
            max_rounds: Maximum refinement rounds per worker (loaded from .env if None)
            log_dir: Directory for logs (creates temp if None)
            model_name: OpenAI model to use (loaded from .env if None)
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
            target_platform: Target platform PlatformConfig
            no_cusolver: If True, disables cuSolver library usage
        """
        # Load environment variables
        load_dotenv()

        # Load configuration from environment
        self.num_workers = num_workers or int(os.getenv("NUM_KERNEL_SEEDS", "4"))
        self.max_rounds = max_rounds or int(os.getenv("MAX_REFINEMENT_ROUNDS", "10"))
        self.model_name = model_name or os.getenv(
            "OPENAI_MODEL", "claude-sonnet-4-20250514"
        )
        self.high_reasoning_effort = high_reasoning_effort

        # Initialize provider
        self.provider = None
        try:
            self.provider = get_model_provider(self.model_name, preferred_provider)
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(
                f"Initialized provider '{self.provider.name}' for model '{self.model_name}'"
            )
        except ValueError as e:
            # Will be handled in setup_logging, just store the error for now
            self._provider_error = str(e)

        # Setup logging
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path.cwd() / "triton_kernel_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Normalize to PlatformConfig
        self._platform_config = (
            target_platform if target_platform else get_platform("cuda")
        )
        self.no_cusolver = no_cusolver
        self.test_timeout_s = test_timeout_s
        self._kernel_backend = get_kernel_backend(kernel_backend)
        if not self._kernel_backend.supports_platform(self._platform_config.name):
            raise ValueError(
                f"Backend {kernel_backend!r} does not support "
                f"platform {self._platform_config.name!r}"
            )
        self.experience_store = None
        if enable_experience_memory:
            db_path = experience_db_path or os.getenv(
                "KERNEL_AGENT_EXPERIENCE_DB",
                str(Path.cwd() / ".kernelagent" / "experiences.sqlite3"),
            )
            self.experience_store = SQLiteExperienceStore(db_path)

        # Setup main logger
        self._setup_logging()

        # Initialize prompt manager
        self.prompt_manager = PromptManager(
            target_platform=self._platform_config,
            kernel_backend=self._kernel_backend,
        )

        # Initialize worker manager
        self.manager = WorkerManager(
            num_workers=self.num_workers,
            max_rounds=self.max_rounds,
            log_dir=self.log_dir,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=self.model_name,
            high_reasoning_effort=self.high_reasoning_effort,
            target_platform=self._platform_config.name,
            no_cusolver=self.no_cusolver,
            test_timeout_s=self.test_timeout_s,
            kernel_backend=self._kernel_backend.name,
        )

    def _setup_logging(self):
        """Setup agent logging."""
        log_file = (
            self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("TritonKernelAgent")

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

    def _call_llm(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Call the LLM provider for the configured model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response text
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.model_name}")

        # Add high_reasoning_effort to kwargs if set
        if self.high_reasoning_effort:
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.model_name, messages, **kwargs)
        return response.content

    def _call_llm_detailed(self, messages: list, **kwargs) -> LLMCallResult:
        """Like ``_call_llm`` but surfaces the truncation-detection signals.

        The seed path needs ``completion_tokens`` / ``finish_reason`` to tell a
        malformed response that was cut at the output cap (continue it) from
        one that is simply wrong (repair it). Without that distinction a seed
        truncated mid-bundle was "repaired" by regenerating everything under
        the same cap and truncated again.
        """
        if not self.provider:
            raise RuntimeError(f"No provider available for model {self.model_name}")

        if getattr(self, "high_reasoning_effort", False):
            kwargs["high_reasoning_effort"] = True

        response = self.provider.get_response(self.model_name, messages, **kwargs)
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

    def _generate_test(
        self,
        problem_description: str,
        provided_test_code: str | None = None,
        verify_direction: str | None = None,
    ) -> str:
        """
        Generate test code for the problem using OpenAI API.

        The test must import from 'kernel' module since each worker writes
        the kernel to 'kernel.py' in their working directory.

        Args:
            problem_description: Description of the problem
            provided_test_code: Optional reference test code provided by user
            verify_direction: Optional verification contract. ``forward`` renders
                the forward-only prompt used by KernelBench scoring (no backward
                or gradient comparison) and rejects leftover grad checks.

        Returns:
            Generated test code in standardized format
        """
        # Use LLM provider if available; no mock fallback allowed
        if not self.provider:
            raise RuntimeError(
                "Unable to generate test code: no LLM provider available and mock fallback disabled"
            )
        try:
            self.logger.info(f"Generating test code using {self.model_name}")

            # Create prompt for test generation using template
            prompt = self.prompt_manager.render_test_generation_prompt(
                problem_description=problem_description,
                provided_test_code=provided_test_code,
                verify_direction=verify_direction,
            )

            # Call the LLM, with focused repair attempts if it changes or skips
            # the shared benchmark input factories. Fix 4-1 A3: three attempts;
            # the third fires only where the previous behavior hard-failed the
            # whole problem.
            messages = [{"role": "user", "content": prompt}]
            attempts = 3
            for attempt in range(attempts):
                # Fix 6-2: request the model's resolved budget so the ask and
                # the server cap agree (a literal larger than the cap makes the
                # usage readout and the truncation threshold disagree).
                response_text = self._call_llm(
                    messages,
                    max_tokens=resolve_output_cap(self.model_name, self.provider),
                )
                self.logger.info("Raw test generation response:\n%s", response_text)
                test_code = self._extract_code_from_response(response_text)
                if not test_code:
                    contract_error = ValueError(
                        "No valid code found in LLM response"
                    )
                else:
                    try:
                        if is_forward_only(verify_direction):
                            # KernelBench scores forward output only: a leftover
                            # training-style grad check is a contract failure, not
                            # a kernel defect, so reject it before it runs.
                            validate_forward_only_test(test_code)
                        validate_test_autograd(test_code)
                        if getattr(getattr(self, "_platform_config", None), "name", None) == "musa":
                            validate_test_layout(test_code)
                            # Fix 3-1: the main test is generated before the
                            # kernel, so it must stay signature-agnostic and go
                            # through bind_kernel_call like every other consumer.
                            validate_test_kernel_call_binding(test_code)
                        validate_generated_test_input_contract(
                            problem_description, test_code
                        )
                    except ValueError as exc:
                        contract_error = exc
                    else:
                        self.logger.info(
                            f"Successfully generated test code using {self.model_name}"
                        )
                        return test_code
                if attempt == attempts - 1:
                    raise contract_error
                self.logger.warning(
                    "Generated test failed validity/contract checks; "
                    "retrying after attempt %d/%d: %s",
                    attempt + 1,
                    attempts,
                    contract_error,
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": (
                                "Repair the test. " + str(contract_error) + ". "
                                "Copy get_inputs() and get_init_inputs() exactly from "
                                "Describe2 and call both factories. Return only the "
                                "complete fenced Python test. Preserve the reference "
                                "semantics and all cases/tolerances. "
                                + (
                                    "FORWARD-ONLY: compare forward outputs only. Remove "
                                    "every .backward()/torch.autograd.grad call and every "
                                    "grad_outputs/gradients argument; no leaf-tensor or "
                                    "requires_grad setup is needed."
                                    if is_forward_only(verify_direction)
                                    else
                                    "For backward, recompute the reference with grad "
                                    "enabled and independent leaf inputs; do not "
                                    "differentiate a no_grad result. "
                                    # Fix 4-1 A2: the concrete legal leaf pattern.
                                    "Build independent leaf inputs with "
                                    "move(tensor.detach(), device, gradients=True) from "
                                    "kernel_case_suite — the result is already a fresh "
                                    "leaf, so never assign .requires_grad afterwards; "
                                    "call .requires_grad_() on a fresh leaf instead if needed."
                                )
                            ),
                        },
                    ]
                )

        except Exception as e:
            self.logger.error(f"Error generating test with LLM API: {e}")
            raise


    def _finish_text_retrievals(self, outcome: dict[str, Any]) -> None:
        """Child text-channel events describe the same joint generation result."""
        for event_id in getattr(self, "_last_experience_text_retrieval_ids", []):
            self.experience_store.set_retrieval_outcome(event_id, {
                **outcome, "parent_generation_event_id": self._last_experience_retrieval_id,
                "parent_generation_direction": "joint", "text_only": True,
            })

    def _generate_kernel_seeds(
        self,
        problem_description: str,
        test_code: str,
        num_seeds: int | None = None,
        execution_direction: str = "unknown",
        response_log_dir: Path | None = None,
    ) -> list[str | KernelBundle]:
        """
        Generate initial kernel implementations using OpenAI API.

        Args:
            problem_description: Description of the kernel to generate
            test_code: Test code that the kernel must pass
            num_seeds: Number of kernel variations to generate

        Returns:
            List of kernel implementation strings
        """
        if num_seeds is None:
            num_seeds = self.num_workers

        # Use LLM provider if available
        if self.provider:
            try:
                self.logger.info(
                    f"Generating {num_seeds} kernel seeds using {self.model_name}"
                )

                # Create prompt with Triton guidelines using template
                prompt = self.prompt_manager.render_kernel_generation_prompt(
                    problem_description=problem_description,
                    test_code=test_code,
                    no_cusolver=self.no_cusolver,
                )
                if self.experience_store is not None:
                    memory = generation_memory(
                        self.experience_store, problem_description,
                        direction=execution_direction,
                        platform=self._platform_config.name, backend=self._kernel_backend.name,
                        session_id=response_log_dir.name if response_log_dir else "generation",
                        logger=self.logger,
                    )
                    self._last_experience_retrieval_id = memory.root_event
                    self._last_experience_text_retrieval_ids = memory.text_events
                    experience_context = memory.context
                    seed_context = format_verified_seed_context(memory.seeds)
                    if experience_context:
                        prompt = (
                            f"{prompt}\n\n{experience_context}\n\n"
                            "Use prior experience only when compatible with the current "
                            "contract. Follow the original output format exactly."
                        )
                        self.logger.info(
                            "Added direction-scoped cross-task experience to generation prompt",
                        )
                    if seed_context:
                        prompt = (
                            f"{prompt}\n\n{seed_context}\n\n"
                            "Adapt seeds to the current structured contract and return "
                            "the complete required output format."
                        )
                        self.logger.info(
                            "Added %d verified implementation seeds",
                            len(memory.seeds),
                        )

                kernels = []
                messages = [{"role": "user", "content": prompt}]
                invalid_musa_responses: list[tuple[int, str, str]] = []

                def extract_seed(response_text: str, seed_index: int, label: str):
                    try:
                        payload = self._extract_kernel_payload(
                            response_text, problem_description
                        )
                        # _extract_kernel_payload returns None (rather than
                        # raising) when no FILE block is recognized at all, e.g.
                        # when the model spent its entire budget before emitting
                        # the bundle. Without this the response was never recorded
                        # as invalid and the complete-format repair below was
                        # skipped entirely.
                        if payload is None:
                            raise ValueError(
                                "no FILE block was recognized in the model response"
                            )
                        return payload
                    except ValueError as exc:
                        bundle = extract_kernel_bundle(response_text)
                        recognized = sorted(bundle.files) if bundle else []
                        saved_path = None
                        if response_log_dir is not None:
                            saved_path = response_log_dir / f"{label}.txt"
                            saved_path.write_text(response_text, encoding="utf-8")
                        self.logger.error(
                            "Invalid MUSA kernel seed %d: %s; recognized files=%s; "
                            "raw response saved to %s",
                            seed_index,
                            exc,
                            recognized,
                            saved_path or "(response log directory unavailable)",
                        )
                        invalid_musa_responses.append((seed_index, response_text, str(exc)))
                        return None

                # Use provider's multiple response capability
                # Fix 1+4: request the model's resolved output budget instead of
                # a hardcoded 20000, and continue truncated seeds instead of
                # regenerating them under the same cap.
                max_completion_tokens = resolve_output_cap(self.model_name, self.provider)
                truncation_threshold = max_completion_tokens - TRUNCATION_MARGIN_TOKENS

                def continue_truncated_seed(
                    seed_index: int,
                    first: LLMCallResult,
                ):
                    """Continue a seed cut at the output cap, then extract it.

                    Returns the extracted payload, or None when the response is
                    malformed for a reason other than truncation (or the
                    continuation budget is exhausted). The original text is
                    extended verbatim: the model is told to emit only the
                    remainder, so concatenation restores the full bundle.
                    """
                    text = first.content
                    result = first
                    label = f"seed_{seed_index}_invalid_response"
                    payload = extract_seed(text, seed_index, label)
                    if payload is not None:
                        return payload
                    reason = invalid_musa_responses[-1][2]
                    for cont in range(MAX_CONTINUATIONS):
                        if not looks_truncated(result, reason, truncation_threshold):
                            break
                        self.logger.info(
                            "Kernel seed %d was cut at the output cap "
                            "(completion_tokens=%s, finish_reason=%s); requesting "
                            "continuation %d/%d",
                            seed_index,
                            result.completion_tokens,
                            result.finish_reason,
                            cont + 1,
                            MAX_CONTINUATIONS,
                        )
                        thread = [
                            *messages,
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": CONTINUATION_PROMPT},
                        ]
                        segment = self._call_llm_detailed(
                            thread,
                            max_tokens=max_completion_tokens,
                            temperature=0.2,
                        )
                        if not segment.content or not segment.content.strip():
                            break
                        text = text + segment.content
                        result = segment
                        payload = extract_seed(
                            text,
                            seed_index,
                            f"seed_{seed_index}_continuation_{cont + 1}_invalid_response",
                        )
                        if payload is not None:
                            return payload
                        reason = invalid_musa_responses[-1][2]
                    return None

                if self.provider.supports_multiple_completions():
                    # Provider supports native multiple completions
                    responses = self.provider.get_multiple_responses(
                        self.model_name,
                        messages,
                        n=num_seeds,
                        temperature=0.8,
                        max_tokens=max_completion_tokens,
                        high_reasoning_effort=self.high_reasoning_effort,
                    )

                    for i, response in enumerate(responses):
                        usage = response.usage if isinstance(response.usage, dict) else {}
                        completion_tokens = usage.get("completion_tokens")
                        first = LLMCallResult(
                            content=response.content,
                            completion_tokens=(
                                completion_tokens
                                if isinstance(completion_tokens, int)
                                else None
                            ),
                            finish_reason=getattr(response, "finish_reason", None),
                        )
                        kernel_code = continue_truncated_seed(i, first)
                        if kernel_code:
                            kernels.append(kernel_code)
                        else:
                            self.logger.warning(
                                "Failed to extract code from kernel seed %d; "
                                "response preview: %r",
                                i,
                                response.content[:2000],
                            )
                else:
                    # Provider doesn't support multiple completions, make individual calls
                    for i in range(num_seeds):
                        first = self._call_llm_detailed(
                            messages,
                            max_tokens=max_completion_tokens,
                            temperature=0.8 + (i * 0.1),
                        )
                        kernel_code = continue_truncated_seed(i, first)

                        if kernel_code:
                            kernels.append(kernel_code)
                        else:
                            self.logger.warning(
                                f"Failed to extract code from kernel seed {i}"
                            )

                if (
                    not kernels
                    and self._kernel_backend.name == "musa"
                    and invalid_musa_responses
                ):
                    # One repair is not enough when the model keeps exhausting the
                    # completion budget in the middle of its reasoning: the repair
                    # prompt is itself longer than the seed prompt, so a single
                    # retry often truncates again. Retry the complete-format
                    # repair a bounded number of times before giving up.
                    for attempt in range(1, MUSA_SEED_REPAIR_ATTEMPTS + 1):
                        seed_index, invalid_response, reason = invalid_musa_responses[-1]
                        self.logger.info(
                            "Requesting complete-format repair %d/%d for MUSA kernel seed %d",
                            attempt,
                            MUSA_SEED_REPAIR_ATTEMPTS,
                            seed_index,
                        )
                        repair_messages = [
                            *messages,
                            {"role": "assistant", "content": invalid_response},
                            {
                                "role": "user",
                                "content": (
                                    "Your previously submitted MUSA bundle was REJECTED "
                                    "for this exact reason:\n\n"
                                    + f"{reason}\n\n"
                                    + "Fix precisely that defect and regenerate the complete "
                                    "native MUSA bundle (kernel.py, binding.cpp, kernel.mu, "
                                    "setup.py), each header literally 'FILE: <name>' immediately "
                                    "before its fenced code block. Do NOT reproduce the rejected "
                                    "construct named above. Return no prose, no reasoning, no "
                                    "markdown outside the four FILE blocks."
                                ),
                            },
                        ]
                        repaired_response = self._call_llm(
                            repair_messages,
                            max_tokens=max_completion_tokens,
                            temperature=0.2,
                        )
                        repaired_kernel = extract_seed(
                            repaired_response,
                            seed_index,
                            f"seed_{seed_index}_repair_{attempt}_invalid_response",
                        )
                        if repaired_kernel:
                            kernels.append(repaired_kernel)
                            self.logger.info(
                                "Successfully repaired MUSA kernel seed %d output format "
                                "on repair attempt %d",
                                seed_index,
                                attempt,
                            )
                            break

                if kernels:
                    self.logger.info(
                        f"Successfully generated {len(kernels)} kernel seeds"
                    )
                    return kernels
                else:
                    self.logger.error(
                        "Failed to extract any valid kernels from LLM responses"
                    )
                    raise ValueError("No valid kernel code found in any LLM response")

            except Exception as e:
                self.logger.error(f"Error generating kernels with LLM API: {e}")
                # Fall back to mock implementation

        # Mock kernel generation (fallback)
        if self._kernel_backend.name != "triton":
            raise RuntimeError(
                f"No valid {self._kernel_backend.display_name} bundle was generated"
            )
        self.logger.info(f"Generating {num_seeds} kernel seeds (mock implementation)")

        kernels = []
        for i in range(num_seeds):
            # Simpler mock that still demonstrates the wrapper pattern
            if i == 2:  # Third kernel will pass
                kernel = '''"""
Kernel implementation - working version.
"""

def kernel_function(*args, **kwargs):
    """Wrapper function that handles kernel launch."""
    # Mock implementation that passes tests
    # In real kernels, this would launch a Triton kernel
    return True
'''
            else:
                kernel = f'''"""
Kernel implementation {i + 1}.
"""

def kernel_function(*args, **kwargs):
    """Wrapper function that handles kernel launch."""
    # Mock implementation that fails
    raise NotImplementedError('Mock kernel not implemented')
'''
            kernels.append(kernel)

        return kernels

    def _extract_kernel_payload(
        self, response_text: str, problem_description: str = ""
    ) -> str | KernelBundle | None:
        if self._kernel_backend.name == "musa":
            bundle = extract_kernel_bundle(response_text)
            if bundle is not None:
                bundle.require(self._kernel_backend)
                # A response truncated by the max_tokens cap, or salvaged out of
                # reasoning_content, can still carry all four FILE headers while
                # the kernel.py body is incomplete. require() only checks that the
                # filenames are present, so accept the bundle only when the
                # wrapper really defines the entry point. Raising here routes the
                # response into the existing "repair the complete format" retry
                # instead of aborting before any verification round can run.
                wrapper = bundle.files.get("kernel.py", "")
                if "def kernel_function" not in wrapper:
                    raise ValueError(
                        "kernel.py is missing required kernel_function definition"
                    )
                reason = validate_musa_bundle_semantics(bundle)
                if reason:
                    raise ValueError(reason)
                # Fix 3-1 follow-up: the kernel is now the side that declares
                # the signature, so a parameter that bind_kernel_call cannot
                # fill only surfaces at runtime -- one full verification round
                # (25-30 min) later (P57 round 6: stride_h/padding_w/batch_size
                # on top of the single tensor input). Reject the seed here and
                # let the existing complete-format repair feed the reason back.
                reason = kernel_signature_bindability(wrapper, problem_description)
                if reason:
                    raise ValueError(reason)
            return bundle
        return self._extract_code_from_response(
            response_text,
            prefer_kernel_function=self._has_multiple_tests,
        )

    def generate_kernel(
        self,
        problem_description: str,
        test_code: str | None = None,
        generate_default_test: bool = True,
        execution_direction: str | None = None,
        verify_direction: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate an optimized Triton kernel for the given problem.

        Args:
            problem_description: Description of the kernel to generate
            test_code: Optional additional test code string.
                      Each test should:
                      1. Import the kernel function: from kernel import kernel_function
                      2. Test the kernel and return True/False
                      3. Exit with code 0 on success, 1 on failure
            execution_direction: Trusted forward, backward, or joint target.
                      Without it, only an explicit marker determines direction.
            verify_direction: Optional verification contract for the generated
                      correctness test. ``forward`` verifies forward outputs
                      only (KernelBench scoring); falls back to
                      KERNELAGENT_VERIFY_DIRECTION, then the legacy contract.
            generate_default_test: If True (default), auto-generate a primary
                      test using the LLM. The generated test runs before any
                      provided ``test_code``.

        Returns:
            Dictionary with results including successful kernel
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting kernel generation")
        self._last_experience_retrieval_id = None
        self._last_experience_text_retrieval_ids = []
        self.logger.info(f"Problem: {problem_description[:100]}...")

        # Fix 7-1: forward-only scoring for benchmarks whose metric compares
        # the forward output (KernelBench). Resolved from the caller, then
        # KERNELAGENT_VERIFY_DIRECTION; None keeps the legacy contract.
        resolved_verify_direction = resolve_verify_direction(verify_direction)
        if resolved_verify_direction:
            self.logger.info(
                "Verification contract: %s-only correctness test",
                resolved_verify_direction,
            )

        # Normalize test_code to list[str]
        test_code_list: list[str] = []
        if generate_default_test:
            generated = self._generate_test(
                problem_description, None, verify_direction=resolved_verify_direction
            )
            self.logger.info("Generated default test code using LLM")
            test_code_list.append(generated)
        if test_code is not None:
            self.logger.info("Appending provided test code")
            test_code_list.append(test_code)
        if not test_code_list:
            raise ValueError(
                "No test code: provide test_code or set generate_default_test=True"
            )
        self._has_multiple_tests = len(test_code_list) > 1

        # Log inputs
        import time

        # Add microseconds to ensure unique directory names
        if execution_direction is not None and execution_direction not in {
            "forward", "backward", "joint"
        }:
            raise ValueError("execution_direction must be forward, backward, or joint")
        declared = next(
            (value for source in (*test_code_list, problem_description)
             if (value := _declared_execution_direction(source))),
            None,
        )
        direction = (
            execution_direction
            or declared
            # A forward-only contract must also drive the deterministic suite so
            # its companion test does not demand gradients.
            or (resolved_verify_direction if resolved_verify_direction == "forward" else None)
            or "unknown"
        )
        from .case_suite import suite_test
        reference_tree = _parse_reference_problem(problem_description)
        companion = None
        if reference_tree is not None:
            companion = suite_test(ast.unparse(reference_tree), direction, self._platform_config.name)
            if companion:
                test_code_list.append(companion)
                self._has_multiple_tests = True
                self.logger.info("Added deterministic verification of all explicit input cases")
        direction_source = (
            "caller" if execution_direction else "explicit_marker" if declared
            else "unknown"
        )
        if execution_direction and declared and execution_direction != declared:
            self.logger.warning(
                "Ignoring conflicting test/request direction %s; caller specified %s",
                declared, execution_direction,
            )
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{int(time.time() * 1000000) % 1000000}"
        )
        session_dir = self.log_dir / f"session_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        with open(session_dir / "problem.txt", "w") as f:
            f.write(problem_description)
        for i, test_code in enumerate(test_code_list):
            with open(session_dir / f"test_{i}.py", "w") as f:
                f.write(test_code)

        # Generate kernel seeds (all tests as LLM context, with labels)
        kernel_seeds = self._generate_kernel_seeds(
            problem_description,
            format_test_code_for_llm(test_code_list),
            response_log_dir=session_dir,
            execution_direction=direction,
        )

        # Save seeds
        for i, kernel in enumerate(kernel_seeds):
            if isinstance(kernel, KernelBundle):
                seed_dir = session_dir / f"seed_{i}"
                seed_dir.mkdir()
                for name, content in kernel.files.items():
                    (seed_dir / name).write_text(content, encoding="utf-8")
            else:
                (session_dir / f"seed_{i}.py").write_text(kernel, encoding="utf-8")

        # Run parallel verification with session directory for worker logs
        result = self.manager.run_verification(
            kernel_seeds=kernel_seeds,
            test_code=test_code_list,
            problem_description=problem_description,
            session_log_dir=session_dir,
        )

        # Process results
        if result and result["success"]:
            self.logger.info(f"Success! Worker {result['worker_id']} found solution")

            # Save successful kernel
            payload = result["kernel_code"]
            if isinstance(payload, dict):
                final_dir = session_dir / "final_kernel"
                final_dir.mkdir()
                for name, content in payload.items():
                    (final_dir / name).write_text(content, encoding="utf-8")
            else:
                (session_dir / "final_kernel.py").write_text(payload, encoding="utf-8")

            # Save full result
            with open(session_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

            if self.experience_store is not None:
                payload = result["kernel_code"]
                stored_code = (
                    KernelBundle(payload).render_for_prompt()
                    if isinstance(payload, dict)
                    else payload
                )
                reference_source = reference_source_from_test(test_code_list[0])
                if reference_tree is not None and companion:
                    # Companion verified the actual reference, including factories
                    # that an LLM's primary test may legitimately omit.
                    reference_source = ast.unparse(reference_tree)
                if reference_source is None:
                    self.logger.warning(
                        "Verified test has no extractable reference Model; "
                        "recording experience without a graph identity"
                    )
                experience_source = reference_source or problem_description
                if reference_source and direction in {"backward", "joint"}:
                    experience_source = reference_source + f"\nKERNELAGENT_OPTIMIZATION_TARGET = {direction!r}\n"
                request_signature = build_operator_signature(problem_description)
                signature = build_operator_signature(
                    experience_source, trace_graph=reference_source is not None,
                )
                self.experience_store.add_online(
                    ExperienceRecord(
                        signature=signature,
                        direction=direction,
                        platform=self._platform_config.name,
                        kernel_backend=self._kernel_backend.name,
                        outcome="generated",
                        kernel_code=stored_code,
                        verified=True,
                        case_results=tuple(result.get("case_results") or ()),
                        action="Generated a correctness-verified kernel",
                        environment={
                            **collect_environment(),
                            "model": self.model_name,
                            "execution_direction_source": direction_source,
                            "request_digest": hashlib.sha256(
                                problem_description.encode("utf-8")
                            ).hexdigest(),
                            "request_signature": request_signature.to_dict(),
                            "reference_source_digest": (
                                hashlib.sha256(reference_source.encode("utf-8"))
                                .hexdigest() if reference_source else ""
                            ),
                            "reference_source_kind": (
                                "verified_case_suite_reference" if companion and reference_source
                                else "verified_test_0" if reference_source
                                else "description_fallback"
                            ),
                        },
                    ),
                    session_id=session_dir.name,
                    test_suite_hash=hashlib.sha256(
                        json.dumps(test_code_list, sort_keys=True).encode()
                    ).hexdigest(),
                )
                retrieval_id = getattr(self, "_last_experience_retrieval_id", None)
                if retrieval_id:
                    self.experience_store.set_retrieval_outcome(
                        retrieval_id, {"success": True, "verified": True}
                    )
                self._finish_text_retrievals({"success": True, "verified": True})

            return {
                "success": True,
                "kernel_code": result["kernel_code"],
                "worker_id": result["worker_id"],
                "rounds": result["rounds"],
                "session_dir": str(session_dir),
            }
        else:
            self.logger.warning("No worker found a successful solution")
            if self.experience_store is not None:
                retrieval_id = getattr(self, "_last_experience_retrieval_id", None)
                if retrieval_id:
                    self.experience_store.set_retrieval_outcome(
                        retrieval_id, {"success": False}
                    )
                self._finish_text_retrievals({"success": False})
            return {
                "success": False,
                "message": "Failed to generate working kernel",
                "session_dir": str(session_dir),
            }

    def cleanup(self):
        """Clean up resources."""
        self.manager.cleanup()
