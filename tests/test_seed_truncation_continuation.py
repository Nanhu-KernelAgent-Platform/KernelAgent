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

"""Unit tests: agent seed-generation output-cap truncation continuation.

Fully mocked -- no GPU, no live LLM calls. The seed path used to treat a
response cut at the output cap as "invalid" and spend its repair attempts on
full regenerations under the same cap, which truncated again; these tests pin
the continuation behavior that replaces it.
"""

import pytest
from unittest.mock import Mock

from triton_kernel_agent.agent import TritonKernelAgent
from triton_kernel_agent.kernel_backend import get_kernel_backend
from triton_kernel_agent.llm_truncation import (
    CONTINUATION_PROMPT,
    LLMCallResult,
    MAX_CONTINUATIONS,
    TRUNCATION_MARGIN_TOKENS,
)


FULL_BUNDLE = (
    "FILE: kernel.py\n```python\ndef kernel_function(x): return extension.forward(x)\n```\n\n"
    "FILE: binding.cpp\n```cpp\n#include <torch/extension.h>\ntorch::Tensor forward(torch::Tensor x);\n```\n\n"
    "FILE: kernel.mu\n```cpp\n__global__ void k(float* out, const float* in) { out[0] = in[0]; }\n```\n\n"
    "FILE: setup.py\n```python\nfrom torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
    "from setuptools import setup\nsetup(name='extension', ext_modules=[MUSAExtension(name='extension', "
    "sources=['binding.cpp', 'kernel.mu'])], cmdclass={'build_ext': BuildExtension})\n```\n"
)

# Seed 0 cut at the output cap: complete kernel.py, binding.cpp opened and cut.
CUT_SEED = (
    "FILE: kernel.py\n```python\ndef kernel_function(x): return extension.forward(x)\n```\n\n"
    "FILE: binding.cpp\n```cpp\n#include <torch/extension.h>\n"
    "torch::Tensor forward(torch::Tensor x) {\n"
)

# Continuation tail: the remainder of binding.cpp plus the remaining files.
TAIL = (
    "    return x;\n}\n```\n\n"
    "FILE: kernel.mu\n```cpp\n__global__ void k(float* out, const float* in) { out[0] = in[0]; }\n```\n\n"
    "FILE: setup.py\n```python\nfrom torch_musa.utils.musa_extension import MUSAExtension, BuildExtension\n"
    "from setuptools import setup\nsetup(name='extension', ext_modules=[MUSAExtension(name='extension', "
    "sources=['binding.cpp', 'kernel.mu'])], cmdclass={'build_ext': BuildExtension})\n```\n"
)


def _agent(tmp_path, model="MusaCoder-27B"):
    (tmp_path / "work").mkdir(exist_ok=True)
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.provider = Mock()
    agent.provider.supports_multiple_completions.return_value = False
    agent.model_name = model
    agent.num_workers = 1
    agent.high_reasoning_effort = False
    agent.experience_store = None
    agent.no_cusolver = False
    agent.prompt_manager = Mock()
    agent.prompt_manager.render_kernel_generation_prompt.return_value = "prompt"
    agent.logger = Mock()
    agent._kernel_backend = get_kernel_backend("musa")
    agent._extract_kernel_payload = TritonKernelAgent._extract_kernel_payload.__get__(agent)
    agent._extract_code_from_response = TritonKernelAgent._extract_code_from_response.__get__(agent)
    agent.max_completion_tokens_seen = []
    return agent


def test_truncated_seed_is_continued_not_regenerated(tmp_path):
    agent = _agent(tmp_path)
    cap = 32768
    agent._call_llm_detailed = Mock(side_effect=[
        # cut at the cap: threshold hit, then the continuation completes it
        LLMCallResult(CUT_SEED, cap, "stop"),
        LLMCallResult(TAIL, 120, "stop"),
    ])

    kernels = agent._generate_kernel_seeds(
        "problem", "from kernel import kernel_function", num_seeds=1
    )

    assert len(kernels) == 1
    bundle = kernels[0]
    assert sorted(bundle.files) == ["binding.cpp", "kernel.mu", "kernel.py", "setup.py"]
    assert "torch::Tensor forward" in bundle.files["binding.cpp"]
    # exactly two calls: the truncated seed plus one continuation (no full
    # regeneration, no format repair)
    assert agent._call_llm_detailed.call_count == 2
    continuation = agent._call_llm_detailed.call_args_list[1].args[0]
    assert continuation[-1] == {"role": "user", "content": CONTINUATION_PROMPT}
    assert continuation[1] == {"role": "assistant", "content": CUT_SEED}


def test_non_truncated_invalid_seed_is_not_continued(tmp_path):
    agent = _agent(tmp_path)
    # malformed but far below the cap: no truncation signal -> no continuation
    agent._call_llm_detailed = Mock(return_value=LLMCallResult(
        "this is prose without any FILE block", 500, "stop"
    ))

    with pytest.raises(RuntimeError, match="No valid MUSA bundle"):
        agent._generate_kernel_seeds(
            "problem", "from kernel import kernel_function", num_seeds=1
        )

    # 1 seed call + the bounded complete-format repairs (MUSA), no continuation
    for call in agent._call_llm_detailed.call_args_list:
        thread = call.args[0]
        assert not any(
            message.get("content") == CONTINUATION_PROMPT for message in thread
        )


def test_continuation_budget_is_bounded(tmp_path):
    agent = _agent(tmp_path)
    incomplete = "FILE: kernel.py\n```python\ndef kernel_function(x): return extension.forward(x)\n```\n"
    responses = [
        LLMCallResult(incomplete, 32768, "stop"),
        LLMCallResult("still nothing useful", 32768, "stop"),
        LLMCallResult("again nothing useful", 32768, "stop"),
        LLMCallResult("and more nothing", 32768, "stop"),
        LLMCallResult("and the last repair", 32768, "stop"),
        LLMCallResult("and the final repair", 32768, "stop"),
    ]
    agent._call_llm_detailed = Mock(side_effect=responses)

    with pytest.raises(RuntimeError, match="No valid MUSA bundle"):
        agent._generate_kernel_seeds(
            "problem", "from kernel import kernel_function", num_seeds=1
        )

    # seed call + at most MAX_CONTINUATIONS continuations, then the bounded
    # complete-format repairs
    continuation_threads = [
        call for call in agent._call_llm_detailed.call_args_list
        if any(
            message.get("content") == CONTINUATION_PROMPT for message in call.args[0]
        )
    ]
    assert len(continuation_threads) <= MAX_CONTINUATIONS
