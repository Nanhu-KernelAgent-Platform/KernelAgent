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
import os
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime
import logging
from dotenv import load_dotenv

from .manager import WorkerManager
from .prompt_manager import PromptManager
from utils.providers import BaseProvider, get_model_provider
from triton_kernel_agent.platform_config import PlatformConfig, get_platform
from triton_kernel_agent.worker_util import format_test_code_for_llm
from triton_kernel_agent.kernel_backend import (
    KernelBundle,
    extract_kernel_bundle,
    get_kernel_backend,
)
from triton_kernel_agent.experience import (
    ExperienceRecord,
    SQLiteExperienceStore,
    build_operator_signature,
    collect_environment,
    format_experience_context,
    format_verified_seed_context,
)


def _parse_reference_problem(problem_description: str):
    """Parse plain problem source or the Describe2 section of a combined prompt."""
    heading = re.search(r"(?m)^## Describe2[^\n]*\n", problem_description)
    source = problem_description[heading.end() :] if heading else problem_description
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


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
        elif ast.dump(test_node, include_attributes=False) != ast.dump(
            reference_node, include_attributes=False
        ):
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

    def _generate_test(
        self, problem_description: str, provided_test_code: str | None = None
    ) -> str:
        """
        Generate test code for the problem using OpenAI API.

        The test must import from 'kernel' module since each worker writes
        the kernel to 'kernel.py' in their working directory.

        Args:
            problem_description: Description of the problem
            provided_test_code: Optional reference test code provided by user

        Returns:
            Generated test code in standardized format
        """
        # Use LLM provider if available; no mock fallback allowed
        if not self.provider:
            raise RuntimeError(
                "Unable to generate test code: no LLM provider available and mock fallback disabled"
            )
        # Use LLM provider if available
        if self.provider:
            try:
                self.logger.info(f"Generating test code using {self.model_name}")

                # Create prompt for test generation using template
                prompt = self.prompt_manager.render_test_generation_prompt(
                    problem_description=problem_description,
                    provided_test_code=provided_test_code,
                )

                # Call the LLM, with one focused repair attempt if it changes or skips
                # the shared benchmark input factories.
                messages = [{"role": "user", "content": prompt}]
                for attempt in range(2):
                    response_text = self._call_llm(messages, max_tokens=24000)
                    self.logger.info("Raw test generation response:\n%s", response_text)
                    test_code = self._extract_code_from_response(response_text)
                    if not test_code:
                        contract_error = ValueError(
                            "No valid code found in LLM response"
                        )
                    else:
                        try:
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
                    if attempt == 1:
                        raise contract_error
                    self.logger.warning(
                        "Generated test violated the input contract; retrying once: %s",
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
                                    "complete fenced Python test."
                                ),
                            },
                        ]
                    )

            except Exception as e:
                self.logger.error(f"Error generating test with LLM API: {e}")
                raise

        # Mock test generation (fallback)
        self.logger.info("Generating test code (mock implementation)")

        # If provided test code exists, create a basic wrapper
        if provided_test_code:
            test_code = '''"""
Test for kernel implementation (adapted from provided test).
"""
import torch

def test_kernel():
    """Test the kernel implementation."""
    from kernel import kernel_function

    # Adapted from provided test code
    try:
        # Create test data (standardized format)
        test_input = torch.randn(1024, device='cuda')

        # Call kernel_function as a normal Python function
        result = kernel_function(test_input)

        # Basic validation
        if result is not None:
            print("Test passed!")
            return True
        else:
            print("Test failed: No result returned")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
'''
        else:
            test_code = '''"""
Test for kernel implementation.
"""
import torch

def test_kernel():
    """Test the kernel implementation."""
    from kernel import kernel_function

    # Mock test - replace with actual test logic
    try:
        # Create test data
        test_input = torch.randn(1024, device='cuda')

        # Call kernel_function as a normal Python function
        # (kernel launch logic is handled inside kernel.py)
        result = kernel_function(test_input)

        print("Test passed!")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)
'''
        return test_code

    def _generate_kernel_seeds(
        self,
        problem_description: str,
        test_code: str,
        num_seeds: int | None = None,
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
                    signature = build_operator_signature(problem_description)
                    experiences = self.experience_store.search(
                        signature,
                        platform=self._platform_config.name,
                        kernel_backend=self._kernel_backend.name,
                        limit=3,
                    )
                    experience_context = format_experience_context(experiences)
                    verified_seeds = self.experience_store.search_verified_seeds(
                        signature,
                        platform=self._platform_config.name,
                        kernel_backend=self._kernel_backend.name,
                        limit=2,
                    )
                    seed_context = format_verified_seed_context(verified_seeds)
                    if experience_context:
                        prompt = (
                            f"{prompt}\n\n{experience_context}\n\n"
                            "Use prior experience only when compatible with the current "
                            "contract. Follow the original output format exactly."
                        )
                        self.logger.info(
                            "Added %d relevant cross-task experiences to generation prompt",
                            len(experiences),
                        )
                    if seed_context:
                        prompt = (
                            f"{prompt}\n\n{seed_context}\n\n"
                            "Adapt seeds to the current structured contract and return "
                            "the complete required output format."
                        )
                        self.logger.info(
                            "Added %d verified implementation seeds",
                            len(verified_seeds),
                        )

                kernels = []
                messages = [{"role": "user", "content": prompt}]
                invalid_musa_responses: list[tuple[int, str]] = []

                def extract_seed(response_text: str, seed_index: int, label: str):
                    try:
                        return self._extract_kernel_payload(response_text)
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
                        invalid_musa_responses.append((seed_index, response_text))
                        return None

                # Use provider's multiple response capability
                max_completion_tokens = 20000

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
                        kernel_code = extract_seed(
                            response.content, i, f"seed_{i}_invalid_response"
                        )
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
                        response_text = self._call_llm(
                            messages,
                            max_tokens=max_completion_tokens,
                            temperature=0.8 + (i * 0.1),
                        )
                        kernel_code = extract_seed(
                            response_text, i, f"seed_{i}_invalid_response"
                        )

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
                    seed_index, invalid_response = invalid_musa_responses[-1]
                    self.logger.info(
                        "Requesting one complete-format repair for MUSA kernel seed %d",
                        seed_index,
                    )
                    repair_messages = [
                        *messages,
                        {"role": "assistant", "content": invalid_response},
                        {
                            "role": "user",
                            "content": (
                                "Regenerate the complete native MUSA bundle. The prior "
                                "answer was incomplete or used unsupported file headers. "
                                "Return all four files with the literal headers "
                                "FILE: kernel.py, FILE: binding.cpp, FILE: kernel.mu, and "
                                "FILE: setup.py. Put each header immediately before its "
                                "fenced code block and return no prose."
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
                        f"seed_{seed_index}_repair_invalid_response",
                    )
                    if repaired_kernel:
                        kernels.append(repaired_kernel)
                        self.logger.info(
                            "Successfully repaired MUSA kernel seed %d output format",
                            seed_index,
                        )

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

    def _extract_kernel_payload(self, response_text: str) -> str | KernelBundle | None:
        if self._kernel_backend.name == "musa":
            bundle = extract_kernel_bundle(response_text)
            if bundle is not None:
                bundle.require(self._kernel_backend)
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
            generate_default_test: If True (default), auto-generate a primary
                      test using the LLM. The generated test runs before any
                      provided ``test_code``.

        Returns:
            Dictionary with results including successful kernel
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting kernel generation")
        self.logger.info(f"Problem: {problem_description[:100]}...")

        # Normalize test_code to list[str]
        test_code_list: list[str] = []
        if generate_default_test:
            generated = self._generate_test(problem_description, None)
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
                self.experience_store.add(
                    ExperienceRecord(
                        signature=build_operator_signature(problem_description),
                        platform=self._platform_config.name,
                        kernel_backend=self._kernel_backend.name,
                        outcome="generated",
                        kernel_code=stored_code,
                        verified=True,
                        action="Generated a correctness-verified kernel",
                        lesson="Reuse only after validating shape, dtype, and device constraints.",
                        environment={
                            **collect_environment(),
                            "model": self.model_name,
                        },
                    )
                )

            return {
                "success": True,
                "kernel_code": result["kernel_code"],
                "worker_id": result["worker_id"],
                "rounds": result["rounds"],
                "session_dir": str(session_dir),
            }
        else:
            self.logger.warning("No worker found a successful solution")
            return {
                "success": False,
                "message": "Failed to generate working kernel",
                "session_dir": str(session_dir),
            }

    def cleanup(self):
        """Clean up resources."""
        self.manager.cleanup()
